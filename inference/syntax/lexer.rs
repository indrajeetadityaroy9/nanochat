//! The lexer: CPython 3.12.3's Parser/tokenizer.c as a character-level machine whose states are values, so a prefix
//! may end anywhere; input as compile() sees it ("\r\n", "\r" read as "\n", a final "\n" implied).

use rustc_hash::FxHashMap;

/// A lexeme: an index into Lexicon::lexemes, the terminals a token may be (a soft keyword's two: name and keyword).
pub type Lexeme = u16;
/// A set of lexeme sequences a partial token must begin with: an index into Lexicon::terms.
pub type Terms = u32;
/// The lexemes of the basic terminals (Lexicon::new's `basic`, in its order)
const NAME: Lexeme = 0;
const NUMBER: Lexeme = 1;
const STRING: Lexeme = 2;
const FSTRING_START: Lexeme = 3;
const FSTRING_END: Lexeme = 4;

/// What the lexer emits: lexemes, line ends (NL; the parser decides NEWLINE), a backslash continuation (CONT), and
/// the whitespace at a line's start.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum Ev {
    Lex(Lexeme),
    Nl,
    Cont,
    Space,
    Tab,
    FormFeed,
}

/// A number literal's sub-state: what was read last (tokenizer.c, Number).
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum Num {
    Zero1,
    Zero,
    ZeroU,
    ZeroD,
    ZeroDU,
    Int,
    IntU,
    Frac0,
    Frac,
    FracU,
    ExpLower,
    ExpUpper,
    ExpS,
    ExpD,
    ExpDU,
    X,
    XU,
    XD,
    O,
    OU,
    OD,
    B,
    BU,
    BD,
}

/// A string's kind after its prefix: plain, bytes (ASCII only), f-string, raw f-string.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum Kind {
    S,
    B,
    F,
    Fr,
}

/// What an f-string's literal text waits for after "{", "}", a backslash, "\N", or inside "\N{...}".
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum Pend {
    No,
    Lb,
    Rb,
    Esc,
    N,
    U,
}

/// NONE: a name that spells no keyword, soft keyword or string prefix.
const NONE: u16 = u16::MAX;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum Sub {
    /// between tokens at a line's start (true: after a backslash continuation)
    Bol(bool),
    Sp,
    /// a name: its spelled prefix (a node of Lexicon::spelled_next) or NONE
    Name(u16),
    Num(Num),
    /// operator characters: a prefix of an operator (a node of Lexicon::op_next)
    Op(u16),
    /// a string's first quote(s): n of them
    Q { kind: Kind, quote: u8, n: u8 },
    S { ascii_only: bool, quote: u8, size: u8, escaped: bool, endq: u8 },
    /// the innermost f-string's literal text or format spec
    Fl { pend: Pend, endq: u8 },
    Com,
    /// after a backslash (true: at a line's start)
    Bs(bool),
}

/// An open replacement field: its bracket depth, and whether it is in its format spec.
pub type Field = (i32, bool);

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Frame {
    pub quote: u8,
    pub size: u8,
    pub raw: bool,
    pub fields: Vec<Field>,
}

/// A lexer state: pending UTF-8 bytes, after "\r", the open f-strings (innermost last), the sub-state.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Lex {
    pub pend: [u8; 3],
    pub npend: u8,
    pub cr: bool,
    pub frames: Vec<Frame>,
    pub sub: Sub,
}

impl Lex {
    pub fn start() -> Lex {
        Lex { pend: [0; 3], npend: 0, cr: false, frames: Vec::new(), sub: Sub::Bol(false) }
    }
}

type Step = Option<(Vec<Frame>, Sub)>;

const OPEN: &[u8] = b"([{";
const CLOSE: &[u8] = b")]}";
/// tokenizer.c tok_get_normal_mode: b, r, u, f with u alone, f and b exclusive, any case
const STRING_PREFIXES: [&str; 8] = ["r", "u", "b", "br", "rb", "f", "fr", "rf"];

/// The lexical tables: lexemes and terminals, tries of spelled names and operators, partial-token beginnings.
pub struct Lexicon {
    /// lexeme -> its terminals
    pub lexemes: Vec<Vec<u16>>,
    /// lexeme -> bracket depth change
    pub depth_change: Vec<i8>,
    spelled_next: Vec<[u16; 128]>,
    spelled_first: [u16; 128],
    spelled_lexeme: Vec<Lexeme>,
    spelled_kind: Vec<Option<Kind>>,
    spelled_terms: Vec<Terms>,
    op_next: Vec<[u16; 128]>,
    op_first: [u16; 128],
    op_extensible: Vec<bool>,
    op_lexeme: Vec<Option<Lexeme>>,
    op_terms: Vec<Terms>,
    op_dot: u16,
    op_dotdot: u16,
    lex_lbrace: Lexeme,
    lex_rbrace: Lexeme,
    lex_dot: Lexeme,
    /// sets of lexeme sequences, each sorted
    pub terms: Vec<Vec<Vec<Lexeme>>>,
    terms_empty: Terms,
    terms_name: Terms,
    terms_number: Terms,
    terms_string: Terms,
    terms_fstring: Terms,
    /// non-ASCII code point ranges (inclusive) of str.isidentifier() (a start) and of ("a" + c).isidentifier()
    xid_start: Vec<(u32, u32)>,
    xid_continue: Vec<(u32, u32)>,
}

fn intern(terms: &mut Vec<Vec<Vec<Lexeme>>>, index: &mut FxHashMap<Vec<Vec<Lexeme>>, Terms>,
          mut set: Vec<Vec<Lexeme>>) -> Terms {
    set.sort();
    set.dedup();
    *index.entry(set.clone()).or_insert_with(|| {
        terms.push(set);
        (terms.len() - 1) as Terms
    })
}

/// The prefixes of words as a trie over ASCII: (prefix strings, next[id][c], first[c]).
fn prefix_trie(words: &[String]) -> (Vec<String>, Vec<[u16; 128]>, [u16; 128]) {
    let mut prefixes: Vec<String> = Vec::new();
    let mut id: FxHashMap<String, u16> = FxHashMap::default();
    for w in words {
        for i in 1..=w.len() {
            let p = &w[..i];
            if !id.contains_key(p) {
                id.insert(p.to_string(), prefixes.len() as u16);
                prefixes.push(p.to_string());
            }
        }
    }
    let mut next = vec![[NONE; 128]; prefixes.len()];
    let mut first = [NONE; 128];
    for (i, p) in prefixes.iter().enumerate() {
        for c in 0u8..128 {
            let mut q = p.clone();
            q.push(c as char);
            if let Some(&j) = id.get(&q) {
                next[i][c as usize] = j;
            }
        }
        if p.len() == 1 {
            first[p.as_bytes()[0] as usize] = i as u16;
        }
    }
    (prefixes, next, first)
}

fn in_ranges(ranges: &[(u32, u32)], c: u32) -> bool {
    let i = ranges.partition_point(|&(_, hi)| hi < c);
    i < ranges.len() && ranges[i].0 <= c
}

/// The first code point of ranges within [lo, hi].
fn first_in(ranges: &[(u32, u32)], lo: u32, hi: u32) -> Option<u32> {
    let i = ranges.partition_point(|&(_, h)| h < lo);
    (i < ranges.len() && ranges[i].0 <= hi).then(|| ranges[i].0.max(lo))
}

impl Lexicon {
    /// The lexicon from keywords, soft keywords and operators with their terminals, the basic terminals, and the
    /// identifier classes.
    pub fn new(keywords: Vec<(String, Vec<u16>)>, soft: Vec<(String, Vec<u16>)>, operators: Vec<(String, Vec<u16>)>,
               basic: [u16; 5], xid_start: Vec<(u32, u32)>, xid_continue: Vec<(u32, u32)>) -> Lexicon {
        let mut lexemes: Vec<Vec<u16>> = basic.iter().map(|&t| vec![t]).collect();
        let mut word_lexeme: FxHashMap<String, Lexeme> = FxHashMap::default();
        for (w, ts) in keywords.iter().chain(&soft) {
            word_lexeme.insert(w.clone(), lexemes.len() as Lexeme);
            lexemes.push(ts.clone());
        }
        let op_words: Vec<String> = operators.iter().map(|(w, _)| w.clone()).collect();
        let mut op_lex: FxHashMap<String, Lexeme> = FxHashMap::default();
        for (w, ts) in &operators {
            op_lex.insert(w.clone(), lexemes.len() as Lexeme);
            lexemes.push(ts.clone());
        }
        let mut depth_change = vec![0i8; lexemes.len()];
        for (i, &c) in OPEN.iter().enumerate() {
            depth_change[op_lex[&(c as char).to_string()] as usize] = 1;
            depth_change[op_lex[&(CLOSE[i] as char).to_string()] as usize] = -1;
        }

        let mut terms = Vec::new();
        let mut index = FxHashMap::default();
        let terms_empty = intern(&mut terms, &mut index, vec![]);
        let terms_name = intern(&mut terms, &mut index, vec![vec![NAME]]);
        let terms_number = intern(&mut terms, &mut index, vec![vec![NUMBER]]);
        let terms_string = intern(&mut terms, &mut index, vec![vec![STRING]]);
        let terms_fstring = intern(&mut terms, &mut index, vec![vec![FSTRING_START]]);

        // words whose spelling decides a token: keywords, soft keywords, string prefixes (any case)
        let mut spelled_words: Vec<String> = keywords.iter().chain(&soft).map(|(w, _)| w.clone()).collect();
        for p in STRING_PREFIXES {
            for m in 0..1u32 << p.len() {
                spelled_words.push(p.chars().enumerate()
                    .map(|(i, c)| if m >> i & 1 == 1 { c.to_ascii_uppercase() } else { c }).collect());
            }
        }
        let (spelled, spelled_next, spelled_first) = prefix_trie(&spelled_words);
        let mut spelled_lexeme = Vec::new();
        let mut spelled_kind = Vec::new();
        let mut spelled_terms = Vec::new();
        for text in &spelled {
            spelled_lexeme.push(word_lexeme.get(text).copied().unwrap_or(NAME));
            let low = text.to_lowercase();
            spelled_kind.push(STRING_PREFIXES.contains(&low.as_str()).then(|| {
                if low.contains('f') {
                    if low.contains('r') { Kind::Fr } else { Kind::F }
                } else if low.contains('b') {
                    Kind::B
                } else {
                    Kind::S
                }
            }));
            let mut set = vec![vec![NAME]];
            set.extend(word_lexeme.iter().filter(|(w, _)| w.starts_with(text.as_str())).map(|(_, &l)| vec![l]));
            set.extend(STRING_PREFIXES.iter().filter(|p| p.starts_with(low.as_str()))
                .map(|p| vec![if p.contains('f') { FSTRING_START } else { STRING }]));
            spelled_terms.push(intern(&mut terms, &mut index, set));
        }

        // operator prefixes, ".." of "..." included
        let (ops, op_next, op_first) = prefix_trie(&op_words);
        let op_extensible: Vec<bool> = ops.iter()
            .map(|p| ops.iter().any(|q| q != p && q.starts_with(p.as_str()))).collect();
        let op_lexeme: Vec<Option<Lexeme>> = ops.iter().map(|p| op_lex.get(p).copied()).collect();
        let op_id = |s: &str| ops.iter().position(|p| p == s).unwrap() as u16;
        let (op_dot, op_dotdot) = (op_id("."), op_id(".."));
        let lex_dot = op_lex["."];
        let mut op_terms = Vec::new();
        for text in &ops {
            let mut set: Vec<Vec<Lexeme>> = op_words.iter().filter(|op| op.starts_with(text.as_str()))
                .map(|op| vec![op_lex[op]]).collect();
            if text == "." {
                set.push(vec![NUMBER]);
            }
            if text == ".." {
                set.push(vec![lex_dot, lex_dot]);
            }
            op_terms.push(intern(&mut terms, &mut index, set));
        }
        Lexicon {
            depth_change,
            spelled_next, spelled_first, spelled_lexeme, spelled_kind, spelled_terms,
            op_next, op_first, op_extensible, op_lexeme, op_terms, op_dot, op_dotdot,
            lex_lbrace: op_lex["{"], lex_rbrace: op_lex["}"], lex_dot,
            terms, terms_empty, terms_name, terms_number, terms_string, terms_fstring,
            xid_start, xid_continue, lexemes,
        }
    }

    fn op_lexeme(&self, op: u16) -> Lexeme {
        self.op_lexeme[op as usize].expect("an operator")
    }

    /// The lexer state after character c (events appended to ev), None if no module continues this way.
    fn char(&self, frames: Vec<Frame>, sub: Sub, c: char, ev: &mut Vec<Ev>) -> Step {
        match sub {
            Sub::Sp => self.start(frames, c, ev, false),
            Sub::Bol(_) => self.start(frames, c, ev, true),
            Sub::Name(text) => {
                if c.is_ascii() {
                    if c.is_ascii_alphanumeric() || c == '_' {
                        let next = if text != NONE { self.spelled_next[text as usize][c as usize] } else { NONE };
                        return Some((frames, Sub::Name(next)));
                    }
                    if (c == '\'' || c == '"') && text != NONE
                        && let Some(kind) = self.spelled_kind[text as usize]
                    {
                        return Some((frames, Sub::Q { kind, quote: c as u8, n: 1 }));
                    }
                } else if in_ranges(&self.xid_continue, c as u32) {
                    return Some((frames, Sub::Name(NONE)));
                }
                ev.push(Ev::Lex(if text == NONE { NAME } else { self.spelled_lexeme[text as usize] }));
                self.start(frames, c, ev, false)
            }
            Sub::Num(state) => self.number_char(frames, state, c, ev),
            Sub::Op(text) => self.operator_char(frames, text, c, ev),
            Sub::Q { kind, quote, n } => self.quote_char(frames, kind, quote, n, c, ev),
            Sub::S { .. } => string_char(frames, sub, c, ev),
            Sub::Fl { .. } => self.literal_char(frames, sub, c, ev),
            Sub::Com => {
                if c == '\n' {
                    ev.push(Ev::Nl);
                    return Some((frames, Sub::Bol(false)));
                }
                Some((frames, sub))
            }
            // after a backslash, only a line end continues (tok_continuation_line)
            Sub::Bs(bol) => {
                if c != '\n' {
                    return None;
                }
                if bol {
                    ev.push(Ev::Cont);
                    return Some((frames, Sub::Bol(true)));
                }
                Some((frames, Sub::Sp))
            }
        }
    }

    /// Character c between tokens (bol: at the start of a line, where whitespace is indentation).
    fn start(&self, frames: Vec<Frame>, c: char, ev: &mut Vec<Ev>, bol: bool) -> Step {
        match c {
            ' ' | '\t' | '\x0c' => {
                if bol {
                    ev.push(match c {
                        ' ' => Ev::Space,
                        '\t' => Ev::Tab,
                        _ => Ev::FormFeed,
                    });
                    return Some((frames, Sub::Bol(false)));
                }
                return Some((frames, Sub::Sp));
            }
            '\n' => {
                ev.push(Ev::Nl);
                return Some((frames, Sub::Bol(false)));
            }
            '#' => return Some((frames, Sub::Com)),
            '\\' => return Some((frames, Sub::Bs(bol))),
            _ => {}
        }
        if c.is_ascii() {
            if c.is_ascii_alphabetic() || c == '_' {
                return Some((frames, Sub::Name(self.spelled_first[c as usize])));
            }
            if c.is_ascii_digit() {
                return Some((frames, Sub::Num(if c == '0' { Num::Zero1 } else { Num::Int })));
            }
            if c == '\'' || c == '"' {
                return Some((frames, Sub::Q { kind: Kind::S, quote: c as u8, n: 1 }));
            }
            if self.op_first[c as usize] != NONE {
                return self.operator_start(frames, c, ev);
            }
            return None;
        }
        if in_ranges(&self.xid_start, c as u32) {
            return Some((frames, Sub::Name(NONE)));
        }
        None
    }

    /// An operator's first character: brackets and a field's ':' and '}' act at once (tokenizer.c's f-string check).
    fn operator_start(&self, mut frames: Vec<Frame>, c: char, ev: &mut Vec<Ev>) -> Step {
        let field = frames.last().and_then(|f| f.fields.last()).map(|f| f.0);
        let op = self.op_first[c as usize];
        let b = c as u8;
        if field == Some(0) && (c == ':' || c == '}') {
            ev.push(Ev::Lex(self.op_lexeme(op)));
            let fields = &mut frames.last_mut().unwrap().fields;
            if c == ':' {
                *fields.last_mut().unwrap() = (0, true);
            } else {
                fields.pop();
            }
            return Some((frames, Sub::Fl { pend: Pend::No, endq: 0 }));
        }
        // brackets extend no operator
        let open = OPEN.contains(&b);
        if let Some(depth) = field
            && (open || CLOSE.contains(&b))
        {
            if depth == 0 && !open {
                return None;
            }
            *frames.last_mut().unwrap().fields.last_mut().unwrap() = (depth + if open { 1 } else { -1 }, false);
        } else if self.op_extensible[op as usize] {
            return Some((frames, Sub::Op(op)));
        }
        ev.push(Ev::Lex(self.op_lexeme(op)));
        Some((frames, Sub::Sp))
    }

    /// Character c after the operator characters text.
    fn operator_char(&self, frames: Vec<Frame>, text: u16, c: char, ev: &mut Vec<Ev>) -> Step {
        if text == self.op_dot && c.is_ascii_digit() {
            return Some((frames, Sub::Num(Num::Frac)));
        }
        let extended = if c.is_ascii() { self.op_next[text as usize][c as usize] } else { NONE };
        if extended != NONE {
            if self.op_extensible[extended as usize] {
                return Some((frames, Sub::Op(extended)));
            }
            ev.push(Ev::Lex(self.op_lexeme(extended)));
            return Some((frames, Sub::Sp));
        }
        if text == self.op_dotdot {
            // two dots: one '.', then '.' again (tokenizer.c backs up)
            ev.push(Ev::Lex(self.lex_dot));
            return self.operator_char(frames, self.op_dot, c, ev);
        }
        ev.push(Ev::Lex(self.op_lexeme(text)));
        self.start(frames, c, ev, false)
    }

    /// Character c in a number literal (tokenizer.c, Number): the state names what was read last.
    fn number_char(&self, frames: Vec<Frame>, state: Num, c: char, ev: &mut Vec<Ev>) -> Step {
        use Num::*;
        let digit = c.is_ascii_digit();
        // Ok(Some(next)), Ok(None): the number may end here; Err: no module continues
        let nxt: Result<Option<Num>, ()> = match state {
            Zero1 | Zero => Ok(if state == Zero1 && "xXoObB".contains(c) {
                Some(match c.to_ascii_lowercase() {
                    'x' => X,
                    'o' => O,
                    _ => B,
                })
            } else if c == '0' {
                Some(Zero)
            } else if digit {
                Some(ZeroD)
            } else if c == '_' {
                Some(ZeroU)
            } else {
                None
            }),
            ZeroU => if digit { Ok(Some(if c == '0' { Zero } else { ZeroD })) } else { Err(()) },
            // leading zeros: only a fraction, exponent or imaginary part saves it
            ZeroD => if digit {
                Ok(Some(ZeroD))
            } else if c == '_' {
                Ok(Some(ZeroDU))
            } else if ".eEjJ".contains(c) {
                Ok(None)
            } else {
                Err(())
            },
            ZeroDU | IntU | FracU | ExpDU | ExpS => if digit {
                Ok(Some(match state {
                    ZeroDU => ZeroD,
                    IntU => Int,
                    FracU => Frac,
                    _ => ExpD,
                }))
            } else {
                Err(())
            },
            Int => Ok(if digit { Some(Int) } else if c == '_' { Some(IntU) } else { None }),
            Frac0 | Frac => Ok(if digit {
                Some(Frac)
            } else if c == '_' && state == Frac {
                Some(FracU)
            } else {
                None
            }),
            ExpLower | ExpUpper => {
                if c == '+' || c == '-' {
                    return Some((frames, Sub::Num(ExpS)));
                }
                if digit {
                    return Some((frames, Sub::Num(ExpD)));
                }
                if !(c.is_ascii_alphabetic() || c == '_') {
                    return None;
                }
                // verify_end_of_number: 'e' may start "else" after a number; the parser judges the name
                ev.push(Ev::Lex(NUMBER));
                let (frames, sub) = self.start(frames, if state == ExpLower { 'e' } else { 'E' }, ev, false)?;
                return self.char(frames, sub, c, ev);
            }
            ExpD => Ok(if digit { Some(ExpD) } else if c == '_' { Some(ExpDU) } else { None }),
            X | XU | XD => {
                if c.is_ascii_hexdigit() {
                    return Some((frames, Sub::Num(XD)));
                }
                if c == '_' && state != XU {
                    Ok(Some(XU))
                } else if state == XD {
                    Ok(None)
                } else {
                    Err(())
                }
            }
            O | OU | OD | B | BU | BD => {
                let octal = matches!(state, O | OU | OD);
                if if octal { ('0'..='7').contains(&c) } else { c == '0' || c == '1' } {
                    return Some((frames, Sub::Num(if octal { OD } else { BD })));
                }
                if digit {
                    return None;
                }
                let under = matches!(state, OU | BU);
                if c == '_' && !under {
                    Ok(Some(if octal { OU } else { BU }))
                } else if matches!(state, OD | BD) {
                    Ok(None)
                } else {
                    Err(())
                }
            }
        };
        match nxt {
            Err(()) => return None,
            Ok(Some(n)) => return Some((frames, Sub::Num(n))),
            Ok(None) => {}
        }
        if matches!(state, Zero1 | Zero | ZeroD | Int | Frac0 | Frac | ExpD) {
            if c == '.' {
                if matches!(state, Frac0 | Frac | ExpD) {
                    ev.push(Ev::Lex(NUMBER));
                    return self.start(frames, c, ev, false);
                }
                return Some((frames, Sub::Num(Frac0)));
            }
            if (c == 'e' || c == 'E') && state != ExpD {
                return Some((frames, Sub::Num(if c == 'e' { ExpLower } else { ExpUpper })));
            }
            if c == 'j' || c == 'J' {
                ev.push(Ev::Lex(NUMBER));
                return Some((frames, Sub::Sp));
            }
        }
        ev.push(Ev::Lex(NUMBER));
        self.start(frames, c, ev, false)
    }

    /// Character c after a string's first quote(s): one, or two (an empty string or a triple quote's start).
    fn quote_char(&self, mut frames: Vec<Frame>, kind: Kind, quote: u8, n: u8, c: char, ev: &mut Vec<Ev>) -> Step {
        let fstring = matches!(kind, Kind::F | Kind::Fr);
        let is_quote = c == quote as char;
        if is_quote && n == 1 {
            return Some((frames, Sub::Q { kind, quote, n: 2 }));
        }
        if is_quote {
            if fstring {
                ev.push(Ev::Lex(FSTRING_START));
                frames.push(Frame { quote, size: 3, raw: kind == Kind::Fr, fields: Vec::new() });
                return Some((frames, Sub::Fl { pend: Pend::No, endq: 0 }));
            }
            return Some((frames, Sub::S { ascii_only: kind == Kind::B, quote, size: 3, escaped: false, endq: 0 }));
        }
        if n == 2 {
            // an empty string, then c
            ev.push(Ev::Lex(if fstring { FSTRING_START } else { STRING }));
            if fstring {
                ev.push(Ev::Lex(FSTRING_END));
            }
            return self.start(frames, c, ev, false);
        }
        if fstring {
            ev.push(Ev::Lex(FSTRING_START));
            frames.push(Frame { quote, size: 1, raw: kind == Kind::Fr, fields: Vec::new() });
            return self.literal_char(frames, Sub::Fl { pend: Pend::No, endq: 0 }, c, ev);
        }
        string_char(frames, Sub::S { ascii_only: kind == Kind::B, quote, size: 1, escaped: false, endq: 0 }, c, ev)
    }

    /// Character c in an f-string's literal text or format spec (tokenizer.c tok_get_fstring_mode).
    fn literal_char(&self, mut frames: Vec<Frame>, sub: Sub, c: char, ev: &mut Vec<Ev>) -> Step {
        let Sub::Fl { pend, endq } = sub else { unreachable!() };
        let lit = |frames: Vec<Frame>, pend: Pend| Some((frames, Sub::Fl { pend, endq: 0 }));
        let frame = frames.last().unwrap();
        let (quote, size, raw) = (frame.quote as char, frame.size, frame.raw);
        let spec = !frame.fields.is_empty();
        match pend {
            // "{{" is a brace, "{" anything else opens a replacement field
            Pend::Lb => {
                if c == '{' {
                    return lit(frames, Pend::No);
                }
                ev.push(Ev::Lex(self.lex_lbrace));
                frames.last_mut().unwrap().fields = vec![(0, false)];
                return self.start(frames, c, ev, false);
            }
            // "}}" is a brace; a single '}' is an error
            Pend::Rb => return if c == '}' { lit(frames, Pend::No) } else { None },
            // a backslash escapes anything but a brace
            Pend::Esc => {
                if c == '{' || c == '}' {
                    return self.literal_char(frames, Sub::Fl { pend: Pend::No, endq: 0 }, c, ev);
                }
                return lit(frames, if c == 'N' && !raw { Pend::N } else { Pend::No });
            }
            Pend::N => {
                if c == '{' {
                    return lit(frames, Pend::U);
                }
                return self.literal_char(frames, Sub::Fl { pend: Pend::No, endq: 0 }, c, ev);
            }
            // \N{...}: '}' ends the name
            Pend::U => {
                if c == '}' {
                    return lit(frames, Pend::No);
                }
                if c != quote && !(c == '\n' && size == 1) {
                    return lit(frames, Pend::U);
                }
            }
            Pend::No => {}
        }
        if c == quote {
            if size == 1 || endq == 2 {
                ev.push(Ev::Lex(FSTRING_END));
                frames.pop();
                return Some((frames, Sub::Sp));
            }
            return Some((frames, Sub::Fl { pend: Pend::No, endq: endq + 1 }));
        }
        if c == '\n' && size == 1 {
            if !spec {
                return None;
            }
            // a format spec ends at a line end; the field continues as an expression
            *frames.last_mut().unwrap().fields.last_mut().unwrap() = (0, false);
            return self.start(frames, c, ev, false);
        }
        if c == '{' {
            if !spec {
                return lit(frames, Pend::Lb);
            }
            ev.push(Ev::Lex(self.lex_lbrace));
            frames.last_mut().unwrap().fields.push((0, false));
            return Some((frames, Sub::Sp));
        }
        if c == '}' {
            if !spec {
                return lit(frames, Pend::Rb);
            }
            ev.push(Ev::Lex(self.lex_rbrace));
            frames.last_mut().unwrap().fields.pop();
            return lit(frames, Pend::No);
        }
        lit(frames, if c == '\\' { Pend::Esc } else { Pend::No })
    }

    /// The state after byte b of UTF-8 text (events appended to ev); None if no module continues (also invalid UTF-8,
    /// NUL).
    pub fn byte(&self, state: &Lex, b: u8, ev: &mut Vec<Ev>) -> Option<Lex> {
        let c = if state.npend == 0 && b < 0x80 {
            b as char
        } else {
            let n = state.npend as usize;
            let mut data = [0u8; 4];
            data[..n].copy_from_slice(&state.pend[..n]);
            data[n] = b;
            match std::str::from_utf8(&data[..n + 1]) {
                Ok(s) => s.chars().next().unwrap(),
                Err(e) if e.error_len().is_none() => {
                    let mut next = state.clone();
                    next.pend[n] = b;
                    next.npend += 1;
                    return Some(next);
                }
                Err(_) => return None,
            }
        };
        if c == '\0' {
            return None;
        }
        if state.cr && c == '\n' {
            return Some(Lex { pend: [0; 3], npend: 0, cr: false, frames: state.frames.clone(), sub: state.sub });
        }
        let (frames, sub) = self.char(state.frames.clone(), state.sub, if c == '\r' { '\n' } else { c }, ev)?;
        Some(Lex { pend: [0; 3], npend: 0, cr: c == '\r', frames, sub })
    }

    /// The most permissive completion of a pending UTF-8 sequence: an identifier start, else continue, else any.
    fn representative(&self, pend: &[u8]) -> char {
        let size = if pend[0] < 0xE0 { 2 } else if pend[0] < 0xF0 { 3 } else { 4 };
        let mut value = (pend[0] & (0x7F >> size)) as u32;
        for &b in &pend[1..] {
            value = value << 6 | (b & 0x3F) as u32;
        }
        let missing = 6 * (size - pend.len()) as u32;
        let lo = (value << missing).max([0x80, 0x800, 0x10000][size - 2]);
        let hi = (value << missing | ((1 << missing) - 1)).min(0x10FFFF);
        let any = if (0xD800..0xE000).contains(&lo) { 0xE000 } else { lo };
        let c = first_in(&self.xid_start, lo, hi).or_else(|| first_in(&self.xid_continue, lo, hi)).unwrap_or(any);
        char::from_u32(c).unwrap()
    }

    /// The pending UTF-8 character's events (by its representative) and the lexeme sequences the partial token must
    /// begin with; None if no character completes it.
    pub fn partial(&self, state: &Lex, ev: &mut Vec<Ev>) -> Option<Terms> {
        let sub = if state.npend > 0 {
            let c = self.representative(&state.pend[..state.npend as usize]);
            self.char(state.frames.clone(), state.sub, c, ev)?.1
        } else {
            state.sub
        };
        Some(match sub {
            Sub::Name(text) => if text != NONE { self.spelled_terms[text as usize] } else { self.terms_name },
            Sub::Num(_) => self.terms_number,
            Sub::Op(text) => self.op_terms[text as usize],
            Sub::Q { kind, .. } => {
                if matches!(kind, Kind::F | Kind::Fr) { self.terms_fstring } else { self.terms_string }
            }
            Sub::S { .. } => self.terms_string,
            _ => self.terms_empty,
        })
    }
}

/// Character c in a string literal's body (tokenizer.c letter_quote); bytes hold ASCII only.
fn string_char(frames: Vec<Frame>, sub: Sub, c: char, ev: &mut Vec<Ev>) -> Step {
    let Sub::S { ascii_only, quote, size, escaped, endq } = sub else { unreachable!() };
    if ascii_only && !c.is_ascii() {
        return None;
    }
    if escaped {
        return Some((frames, Sub::S { ascii_only, quote, size, escaped: false, endq: 0 }));
    }
    if c == quote as char {
        if size == 1 || endq == 2 {
            ev.push(Ev::Lex(STRING));
            return Some((frames, Sub::Sp));
        }
        return Some((frames, Sub::S { ascii_only, quote, size, escaped: false, endq: endq + 1 }));
    }
    if c == '\n' && size == 1 {
        return None;
    }
    Some((frames, Sub::S { ascii_only, quote, size, escaped: c == '\\', endq: 0 }))
}
