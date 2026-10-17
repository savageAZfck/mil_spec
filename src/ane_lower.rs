//! `ane_lower` — text-level MIL → ANE-eligible MIL lowering.
//!
//! Implements the rules in `ANE_FRONTEND_SPEC.md` that CoreML-emitted
//! stateful models violate: int32 index arithmetic must be baked to
//! constants, and a stripe-partitioned cache state (one
//! `state<tensor<T,[D,..]>>` updated by a chain of `slice_update`s with a
//! single trailing `write_state`) must become one state per stripe, each
//! with its own `read_state` → `slice_update` → `write_state` triple.
//!
//! Operates on the Apple text grammar (`program(1.3)`, `func
//! main<ios18>`): one statement per line, `type name = op(args)[attrs];`.
//! Dynamic index expressions are evaluated with `pos` bound to a
//! concrete value — the caller chooses the decode position to bake.
//!
//! ```text
//! let (text, report) = mil_spec::ane_lower::lower_for_ane(&mil_text, 0)?;
//! ```

use std::collections::{BTreeMap, BTreeSet};

/// What the pass changed, for diagnostics/tests.
#[derive(Debug, Default)]
pub struct LowerReport {
    /// `state` inputs split into per-stripe states.
    pub states_split: usize,
    /// `slice_update` ops localized to a per-stripe state.
    pub updates_localized: usize,
    /// `slice_by_index` readers rebased onto a stripe state.
    pub reads_localized: usize,
    /// `write_state` ops emitted (one per stripe).
    pub writes_emitted: usize,
    /// Index-math ops and unreferenced consts removed.
    pub ops_dropped: usize,
    /// Signature params dropped (e.g. a baked `pos` input).
    pub params_dropped: usize,
}

/// A parsed statement: `type name = op(args)[attrs];`.
/// `write_state` has no binding — `out_name`/`out_ty` are empty.
struct Stmt {
    out_ty: String,
    out_name: String,
    op: String,
    args: Vec<(String, String)>,
    attrs: Vec<(String, String)>,
    line: String,
    line_idx: usize,
}

/// A function-input `state` parameter.
#[derive(Clone)]
struct StateParam {
    name: String,
    dtype: String,
    dims: Vec<i64>,
}

fn split_top(s: &str, sep: char) -> Vec<String> {
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut cur = String::new();
    for c in s.chars() {
        match c {
            '(' | '[' | '<' | '{' => depth += 1,
            ')' | ']' | '>' | '}' => depth -= 1,
            _ => {}
        }
        if c == sep && depth == 0 {
            out.push(cur.trim().to_string());
            cur.clear();
        } else {
            cur.push(c);
        }
    }
    if !cur.trim().is_empty() {
        out.push(cur.trim().to_string());
    }
    out
}

fn kv_list(s: &str) -> Vec<(String, String)> {
    split_top(s, ',')
        .into_iter()
        .filter_map(|p| {
            p.split_once('=')
                .map(|(k, v)| (k.trim().into(), v.trim().into()))
        })
        .collect()
}

/// Parse `tensor<dtype, [d0, d1, ..]>` → (dtype, dims).
fn parse_tensor_ty(s: &str) -> Option<(String, Vec<i64>)> {
    let inner = s.trim().strip_prefix("tensor<")?.strip_suffix('>')?;
    let (dt, rest) = inner.split_once(',')?;
    let dims = rest
        .trim()
        .strip_prefix('[')?
        .strip_suffix(']')?
        .split(',')
        .map(|d| d.trim().parse::<i64>().ok())
        .collect::<Option<Vec<_>>>()?;
    Some((dt.trim().to_string(), dims))
}

/// Parse `state<tensor<dtype,[dims]>> name`.
fn parse_state_param(s: &str) -> Option<StateParam> {
    let s = s.trim();
    let inner = s.strip_prefix("state<")?;
    let gt = inner.rfind('>')?;
    let (dtype, dims) = parse_tensor_ty(&inner[..gt])?;
    let name = inner[gt + 1..].trim().to_string();
    if name.is_empty() {
        return None;
    }
    Some(StateParam { name, dtype, dims })
}

/// Split a statement tail `op(args)[attrs]` into (head, attrs).
/// The attrs `[` is found by backward bracket-stack matching — the attr
/// values themselves contain `[`/`]` (e.g. `tensor<int32, [1]>([0])`).
fn split_attrs(t: &str) -> (&str, Vec<(String, String)>) {
    if !t.ends_with(']') {
        return (t, vec![]);
    }
    let mut stack: Vec<char> = Vec::new();
    let chars: Vec<(usize, char)> = t.char_indices().collect();
    for &(i, c) in chars.iter().rev() {
        match c {
            ']' | ')' | '}' => stack.push(c),
            '[' => {
                if stack.pop() == Some(']') {
                    if stack.is_empty() {
                        return (t[..i].trim_end(), kv_list(&t[i + 1..t.len() - 1]));
                    }
                } else {
                    return (t, vec![]); // unbalanced
                }
            }
            '(' => {
                if stack.pop() != Some(')') {
                    return (t, vec![]);
                }
            }
            '{' if stack.pop() != Some('}') => {
                return (t, vec![]);
            }
            _ => {}
        }
    }
    (t, vec![])
}

/// Bare `op(args)[attrs];` — used for `write_state`.
fn parse_bare(t: &str) -> Option<Stmt> {
    let t = t.trim().trim_end_matches(';');
    let (head, attrs) = split_attrs(t);
    let (op, argstr) = head.strip_suffix(')')?.split_once('(')?;
    let op = op.trim();
    if op.contains(' ') || op.contains('<') || op.is_empty() {
        return None;
    }
    Some(Stmt {
        out_ty: String::new(),
        out_name: String::new(),
        op: op.to_string(),
        args: kv_list(argstr),
        attrs,
        line: String::new(), // filled by caller
        line_idx: 0,         // filled by caller
    })
}

fn parse_stmt(line: &str) -> Option<Stmt> {
    let t = line.trim();
    if t.is_empty() || !t.ends_with(';') || t.contains("func ") || t.contains("-> (") {
        return None;
    }
    // Bare calls like write_state(...) have no `type name =` binding.
    // Detect: text before the first '(' contains no '=' outside parens.
    let head = &t[..t.find('(').unwrap_or(t.len())];
    if !head.contains('=') {
        let mut s = parse_bare(t)?;
        s.line = line.to_string();
        return Some(s);
    }
    let t = t.trim_end_matches(';');
    let (head, attrs) = split_attrs(t);
    let (lhs, rhs) = head.split_once('=')?;
    let (op, argstr) = rhs.trim().strip_suffix(')')?.split_once('(')?;
    let mut parts = lhs.trim().rsplitn(2, ' ');
    let name = parts.next()?.to_string();
    let ty = parts.next().unwrap_or("").to_string();
    Some(Stmt {
        out_ty: ty,
        out_name: name,
        op: op.trim().to_string(),
        args: kv_list(argstr),
        attrs,
        line: line.to_string(),
        line_idx: 0, // filled by caller
    })
}

fn arg<'a>(args: &'a [(String, String)], k: &str) -> Option<&'a str> {
    args.iter().find(|(n, _)| n == k).map(|(_, v)| v.as_str())
}

fn attr<'a>(attrs: &'a [(String, String)], k: &str) -> Option<&'a str> {
    attrs.iter().find(|(n, _)| n == k).map(|(_, v)| v.as_str())
}

/// Constant int tensor literal: `tensor<int32, [n]>([a, b, ..])`.
fn parse_const_vec(s: &str) -> Option<Vec<i64>> {
    let s = s.trim();
    let start = s.rfind('(')?;
    let end = s.rfind(')')?;
    let body = s[start + 1..end]
        .trim()
        .trim_start_matches('[')
        .trim_end_matches(']');
    body.split(',')
        .map(|p| p.trim().parse::<i64>().ok())
        .collect()
}

struct Evaluator<'a> {
    defs: &'a BTreeMap<String, &'a Stmt>,
    env: BTreeMap<String, Vec<i64>>,
    pos: i64,
    int_params: BTreeSet<String>,
}

impl<'a> Evaluator<'a> {
    fn eval(&mut self, var: &str) -> Option<Vec<i64>> {
        let var = var.trim();
        if let Some(v) = self.env.get(var) {
            return Some(v.clone());
        }
        if self.int_params.contains(var) {
            self.env.insert(var.into(), vec![self.pos]);
            return Some(vec![self.pos]);
        }
        let st = *self.defs.get(var)?;
        let v = match st.op.as_str() {
            "const" => parse_const_vec(attr(&st.attrs, "val")?)?,
            "add" | "sub" | "mul" => {
                let x = self.eval(arg(&st.args, "x")?)?;
                let y = self.eval(arg(&st.args, "y")?)?;
                let (x, y) = broadcast(&x, &y)?;
                let f: fn(i64, i64) -> i64 = match st.op.as_str() {
                    "add" => |a, b| a + b,
                    "sub" => |a, b| a - b,
                    _ => |a, b| a * b,
                };
                x.iter().zip(y.iter()).map(|(a, b)| f(*a, *b)).collect()
            }
            "concat" => {
                let vs = arg(&st.args, "values")?;
                let vs = vs.trim().strip_prefix('(')?.strip_suffix(')')?;
                let mut out = Vec::new();
                for part in split_top(vs, ',') {
                    out.extend(self.eval(&part)?);
                }
                out
            }
            _ => return None,
        };
        self.env.insert(var.into(), v.clone());
        Some(v)
    }
}

fn broadcast(x: &[i64], y: &[i64]) -> Option<(Vec<i64>, Vec<i64>)> {
    if x.len() == y.len() {
        Some((x.to_vec(), y.to_vec()))
    } else if x.len() == 1 {
        Some((vec![x[0]; y.len()], y.to_vec()))
    } else if y.len() == 1 {
        Some((x.to_vec(), vec![y[0]; x.len()]))
    } else {
        None
    }
}

fn const_line(name: &str, v: &[i64]) -> String {
    format!(
        "            tensor<int32, [{n}]> {name} = const()[val = tensor<int32, [{n}]>({v:?})];",
        n = v.len()
    )
}

/// One link in a `slice_update` chain over a state's `read_state` value.
struct Stripe {
    su_idx: usize,
    base: i64,
    width: i64,
    begin: Vec<i64>,
    end: Vec<i64>,
}

/// Lower `program(1.x)` MIL text into ANE-frontend-eligible MIL.
///
/// `pos` is the decode position baked into every dynamic index
/// expression — these stateful shards are compiled for a fixed seq
/// length, so a single concrete value is correct.
pub fn lower_for_ane(text: &str, pos: i64) -> Result<(String, LowerReport), String> {
    let mut rep = LowerReport::default();
    let lines: Vec<String> = text.lines().map(|l| l.to_string()).collect();

    // ---- signature ----------------------------------------------------------
    let sig_idx = lines
        .iter()
        .position(|l| l.contains("func main<"))
        .ok_or("no func main<...> signature")?;
    let (sig_head, sig_rest) = lines[sig_idx]
        .split_once('(')
        .ok_or("malformed signature")?;
    let (params_s, sig_close) = sig_rest.split_once(')').ok_or("malformed signature")?;
    let params = split_top(params_s, ',');

    let mut states: Vec<StateParam> = Vec::new();
    let mut int_params = BTreeSet::new();
    for p in &params {
        if let Some(sp) = parse_state_param(p) {
            states.push(sp);
        } else if p.trim().starts_with("tensor<int32,") {
            if let Some(n) = p.trim().rsplit(' ').next() {
                int_params.insert(n.to_string());
            }
        }
    }
    if states.is_empty() {
        return Err("no state<> params to lower".into());
    }

    // ---- statements ----------------------------------------------------------
    let stmt_of_line: Vec<Option<Stmt>> = lines
        .iter()
        .enumerate()
        .map(|(i, l)| {
            parse_stmt(l).map(|mut s| {
                s.line_idx = i;
                s
            })
        })
        .collect();
    let stmts: Vec<&Stmt> = stmt_of_line.iter().flatten().collect();
    let defs: BTreeMap<String, &Stmt> = stmts
        .iter()
        .filter(|s| !s.out_name.is_empty())
        .map(|s| (s.out_name.clone(), *s))
        .collect();

    let mut ev = Evaluator {
        defs: &defs,
        env: BTreeMap::new(),
        pos,
        int_params,
    };

    // read_state var → state name
    let readers: BTreeMap<String, String> = stmts
        .iter()
        .filter(|s| s.op == "read_state")
        .filter_map(|s| arg(&s.args, "input").map(|i| (s.out_name.clone(), i.to_string())))
        .collect();

    // Walk each state's slice_update chain; record dim-0 stripe regions.
    let mut plan: BTreeMap<String, Vec<Stripe>> = BTreeMap::new();
    for s in &stmts {
        if s.op != "slice_update" {
            continue;
        }
        let x = arg(&s.args, "x").unwrap_or("");
        let Some(state) = readers.get(x).cloned() else {
            continue;
        };
        let mut cur: &Stmt = s;
        loop {
            let b = ev
                .eval(arg(&cur.args, "begin").ok_or("slice_update missing begin")?)
                .ok_or_else(|| format!("cannot eval begin of {}", cur.out_name))?;
            let e = ev
                .eval(arg(&cur.args, "end").ok_or("slice_update missing end")?)
                .ok_or_else(|| format!("cannot eval end of {}", cur.out_name))?;
            if b.is_empty() || e.is_empty() || e[0] <= b[0] {
                return Err(format!("empty/invalid update region on {}", cur.out_name));
            }
            plan.entry(state.clone()).or_default().push(Stripe {
                su_idx: cur.line_idx,
                base: b[0],
                width: e[0] - b[0],
                begin: b,
                end: e,
            });
            match stmts.iter().find(|n| {
                n.op == "slice_update" && arg(&n.args, "x") == Some(cur.out_name.as_str())
            }) {
                Some(n) => cur = n,
                None => break,
            }
        }
    }
    if plan.is_empty() {
        return Err("no slice_update chains on read_state values".into());
    }
    // Stripes must tile [0, dim0) exactly.
    for (name, stripes) in &plan {
        let st = states.iter().find(|s| s.name == *name).unwrap();
        let mut bases: Vec<i64> = stripes.iter().map(|s| s.base).collect();
        bases.sort_unstable();
        let mut covered = 0i64;
        for b in &bases {
            if *b != covered {
                return Err(format!(
                    "state {name}: stripes don't tile dim0 (gap at {b})"
                ));
            }
            covered += stripes.iter().find(|s| s.base == *b).unwrap().width;
        }
        if covered != st.dims[0] {
            return Err(format!(
                "state {name}: stripes cover {covered} of {}",
                st.dims[0]
            ));
        }
    }

    // su var → (state, stripe idx, base); su line idx → stripe
    let by_line: BTreeMap<usize, &Stmt> = stmts.iter().map(|s| (s.line_idx, *s)).collect();
    let mut su_home: BTreeMap<String, (String, usize, i64)> = BTreeMap::new();
    for (name, stripes) in &plan {
        for (i, st) in stripes.iter().enumerate() {
            su_home.insert(
                by_line[&st.su_idx].out_name.clone(),
                (name.clone(), i, st.base),
            );
        }
    }
    let split_states: BTreeSet<String> = plan.keys().cloned().collect();

    // ---- rebuild the body ----------------------------------------------------
    let mut emitted: BTreeMap<usize, Vec<String>> = BTreeMap::new();
    let mut drops: BTreeSet<usize> = BTreeSet::new();
    for s in &stmts {
        let i = s.line_idx;
        match s.op.as_str() {
            // Shared read_state / trailing write_state on a split state: gone.
            "read_state" | "write_state"
                if split_states.contains(arg(&s.args, "input").unwrap_or("")) =>
            {
                drops.insert(i);
                rep.ops_dropped += 1;
            }
            "slice_update" if su_home.contains_key(&s.out_name) => {
                let (state, stripe_i, base) = su_home[&s.out_name].clone();
                let st = states.iter().find(|x| x.name == state).unwrap();
                let stripe = &plan[&state][stripe_i];
                let sname = format!("{state}_{stripe_i}");
                let rd = format!("{}_r", s.out_name);
                let mut dims = st.dims.clone();
                dims[0] = stripe.width;
                let mut b = stripe.begin.clone();
                let mut e = stripe.end.clone();
                b[0] -= base;
                e[0] -= base;
                let nb = format!("{}_lb", s.out_name);
                let ne = format!("{}_le", s.out_name);
                let args_s = s
                    .args
                    .iter()
                    .map(|(k, v)| {
                        let v = match k.as_str() {
                            "begin" => nb.clone(),
                            "end" => ne.clone(),
                            "x" => rd.clone(),
                            _ => v.clone(),
                        };
                        format!("{k} = {v}")
                    })
                    .collect::<Vec<_>>()
                    .join(", ");
                emitted.insert(
                    i,
                    vec![
                        const_line(&nb, &b),
                        const_line(&ne, &e),
                        format!(
                            "            tensor<{}, {:?}> {rd} = read_state(input = {sname})[name = string(\"{rd}\")];",
                            st.dtype, dims
                        ),
                        format!(
                            "            tensor<{}, {:?}> {} = slice_update({args_s})[name = string(\"{}\")];",
                            st.dtype, dims, s.out_name, s.out_name
                        ),
                        format!(
                            "            write_state(data = {}, input = {sname})[name = string(\"{}_ws\")];",
                            s.out_name, s.out_name
                        ),
                    ],
                );
                rep.updates_localized += 1;
                rep.writes_emitted += 1;
            }
            "slice_by_index" => {
                let Some((state, stripe_i, base)) =
                    arg(&s.args, "x").and_then(|x| su_home.get(x).cloned())
                else {
                    continue;
                };
                let (Some(mut b), Some(mut e)) = (
                    arg(&s.args, "begin").and_then(|v| ev.eval(v)),
                    arg(&s.args, "end").and_then(|v| ev.eval(v)),
                ) else {
                    return Err(format!(
                        "slice {} on a stripe state has non-const indices",
                        s.out_name
                    ));
                };
                let st = states.iter().find(|x| x.name == state).unwrap();
                let width = plan[&state][stripe_i].width;
                if b.len() != st.dims.len() || e.len() != st.dims.len() {
                    continue;
                }
                if b[0] < base || e[0] > base + width {
                    return Err(format!(
                        "slice {} crosses stripe boundary on {state}",
                        s.out_name
                    ));
                }
                b[0] -= base;
                e[0] -= base;
                let nb = format!("{}_lb", s.out_name);
                let ne = format!("{}_le", s.out_name);
                let args_s = s
                    .args
                    .iter()
                    .map(|(k, v)| {
                        let v = match k.as_str() {
                            "begin" => nb.clone(),
                            "end" => ne.clone(),
                            _ => v.clone(),
                        };
                        format!("{k} = {v}")
                    })
                    .collect::<Vec<_>>()
                    .join(", ");
                let attr_s = if s.attrs.is_empty() {
                    String::new()
                } else {
                    format!(
                        "[{}]",
                        s.attrs
                            .iter()
                            .map(|(k, v)| format!("{k} = {v}"))
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                };
                emitted.insert(
                    i,
                    vec![
                        const_line(&nb, &b),
                        const_line(&ne, &e),
                        format!(
                            "            {} {} = slice_by_index({args_s}){attr_s};",
                            s.out_ty, s.out_name
                        ),
                    ],
                );
                rep.reads_localized += 1;
            }
            _ => {}
        }
    }

    // Assemble: header lines (incl. new signature), rewritten body, tail.
    let first_stmt = stmt_of_line
        .iter()
        .position(|s| s.is_some())
        .unwrap_or(lines.len());
    let last_stmt = stmt_of_line.iter().rposition(|s| s.is_some()).unwrap_or(0);
    let mut new_params: Vec<String> = Vec::new();
    for p in &params {
        if let Some(sp) = parse_state_param(p) {
            if let Some(stripes) = plan.get(&sp.name) {
                for (i, st) in stripes.iter().enumerate() {
                    let mut dims = sp.dims.clone();
                    dims[0] = st.width;
                    new_params.push(format!(
                        "state<tensor<{}, {:?}>> {}_{i}",
                        sp.dtype, dims, sp.name
                    ));
                }
                rep.states_split += 1;
                continue;
            }
        }
        new_params.push(p.clone());
    }
    let mut result: Vec<String> = lines[..first_stmt].to_vec();
    // Rewrite the signature line in-place (it's inside result already).
    for l in result.iter_mut() {
        if l.contains("func main<") {
            *l = format!("{sig_head}({}){}", new_params.join(", "), sig_close);
        }
    }
    for (i, l) in lines
        .iter()
        .enumerate()
        .take(last_stmt + 1)
        .skip(first_stmt)
    {
        if stmt_of_line[i].is_none() {
            result.push(l.clone());
        } else if drops.contains(&i) {
            continue;
        } else if let Some(v) = emitted.get(&i) {
            result.extend(v.iter().cloned());
        } else {
            result.push(l.clone());
        }
    }
    result.extend(lines[last_stmt + 1..].iter().cloned());
    let mut text = result.join("\n");

    // Prune unreferenced int32-producing ops / dead consts to fixpoint.
    loop {
        let used = collect_used_vars(&text);
        let mut removed = false;
        text = text
            .lines()
            .filter(|l| {
                let Some(s) = parse_stmt(l) else { return true };
                if s.out_name.is_empty() || used.contains(&s.out_name) {
                    return true;
                }
                let droppable = s.out_ty.contains("int32") && s.op != "const" || is_dead_const(&s);
                if droppable {
                    removed = true;
                    rep.ops_dropped += 1;
                }
                !droppable
            })
            .collect::<Vec<_>>()
            .join("\n");
        if !removed {
            break;
        }
    }

    // Prune unreferenced signature params (e.g. baked pos).
    let used = collect_used_vars(&text);
    let final_params: Vec<String> = new_params
        .iter()
        .filter(|p| {
            let name = p.trim().rsplit(' ').next().unwrap_or("");
            if !used.contains(name) {
                rep.params_dropped += 1;
                return false;
            }
            true
        })
        .cloned()
        .collect();
    let mut final_lines: Vec<String> = text.lines().map(|l| l.to_string()).collect();
    if let Some(si) = final_lines.iter().position(|l| l.contains("func main<")) {
        let sg = final_lines[si].clone();
        if let Some((h, t)) = sg.split_once('(') {
            if let Some((_ps, c)) = t.split_once(')') {
                final_lines[si] = format!("{h}({}){c}", final_params.join(", "));
            }
        }
    }
    let mut text = final_lines.join("\n");

    // Force ios18 — the only ANE-frontend target we emit.
    if let Some(a) = text.find("main<") {
        if let Some(b) = text[a..].find('>') {
            let tag = &text[a..a + b + 1];
            if tag != "main<ios18>" {
                text = text.replacen(tag, "main<ios18>", 1);
            }
        }
    }
    Ok((text, rep))
}

/// An unreferenced `const` is always safe to drop.
fn is_dead_const(s: &Stmt) -> bool {
    s.op == "const"
}

fn collect_used_vars(text: &str) -> BTreeSet<String> {
    let mut used = BTreeSet::new();
    for l in text.lines() {
        if l.contains("-> (") {
            if let (Some(a), Some(b)) = (l.rfind('('), l.rfind(')')) {
                for t in l[a + 1..b].split(',') {
                    let t = t.trim();
                    if !t.is_empty() {
                        used.insert(t.to_string());
                    }
                }
            }
            continue;
        }
        let Some(s) = parse_stmt(l) else { continue };
        for (_, v) in s.args.iter().chain(s.attrs.iter()) {
            let v = v.trim().trim_start_matches('(').trim_end_matches(')');
            for tok in split_top(v, ',') {
                let t = tok.trim();
                if !t.is_empty()
                    && t.chars()
                        .next()
                        .map(|c| c.is_alphabetic() || c == '_')
                        .unwrap_or(false)
                    && t.chars()
                        .all(|c| c.is_alphanumeric() || c == '_' || c == '.')
                {
                    used.insert(t.to_string());
                }
            }
        }
    }
    used
}
