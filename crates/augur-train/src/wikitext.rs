//! Converts MediaWiki markup into plain prose.
//!
//! This is not a full wikitext parser: it keeps the readable text of
//! articles and drops everything else (templates, tables, references, files,
//! lists, headings and reference sections). Imperfect output only costs a few
//! noisy n-grams, so the rules favour dropping text over keeping junk.

use quick_xml::escape::resolve_html5_entity;

/// Sections whose content is not prose (compared lowercase).
const SKIPPED_SECTIONS: &[&str] = &[
    "references",
    "notes",
    "footnotes",
    "sources",
    "citations",
    "bibliography",
    "further reading",
    "external links",
    "other websites",
    "see also",
    "related pages",
    "gallery",
    "works",
    "discography",
    "filmography",
];

/// HTML-like tags whose content is dropped.
const DROPPED_TAGS: &[&str] = &[
    "ref",
    "references",
    "math",
    "chem",
    "ce",
    "gallery",
    "table",
    "code",
    "syntaxhighlight",
    "source",
    "pre",
    "timeline",
    "score",
    "hiero",
    "imagemap",
    "graph",
    "mapframe",
    "maplink",
    "templatedata",
    "templatestyles",
    "inputbox",
    "categorytree",
    "includeonly",
    "noinclude",
];

/// Link namespaces whose links are dropped (compared lowercase).
const DROPPED_NAMESPACES: &[&str] = &["file", "image", "media", "category"];

/// Templates that render their first positional parameter.
const TEXT_TEMPLATES: &[&str] = &[
    "nowrap",
    "nobr",
    "small",
    "big",
    "nobold",
    "noitalic",
    "smallcaps",
    "em",
    "strong",
    "ill",
    "linktext",
    "vanchor",
];

/// Returns the prose of a wikitext article, one paragraph or line per text
/// line.
pub fn to_plain_text(wikitext: &str) -> String {
    let mut inline = String::with_capacity(wikitext.len() / 2);
    strip_inline(wikitext, &mut inline);
    keep_prose_lines(&inline)
}

/// Whether the article is a disambiguation page.
pub fn is_disambiguation(wikitext: &str) -> bool {
    let lower = wikitext.to_ascii_lowercase();
    ["{{disambig", "{{disambiguation", "{{dab}}", "{{set index"]
        .iter()
        .any(|marker| lower.contains(marker))
}

/// Removes inline markup, appending the visible text to `out`.
fn strip_inline(text: &str, out: &mut String) {
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < text.len() {
        let rest = &text[i..];
        let line_start = i == 0 || bytes[i - 1] == b'\n';
        let consumed = if rest.starts_with("<!--") {
            Some(rest.find("-->").map_or(rest.len(), |end| end + 3))
        } else if rest.starts_with("{{") {
            find_close(rest, "{{", "}}").inspect(|&end| template(&rest[2..end - 2], out))
        } else if line_start && rest.trim_start_matches([' ', '\t']).starts_with("{|") {
            let offset = rest.len() - rest.trim_start_matches([' ', '\t']).len();
            Some(find_close(&rest[offset..], "{|", "|}").map_or(rest.len(), |end| offset + end))
        } else if rest.starts_with("[[") {
            find_close(rest, "[[", "]]").inspect(|&end| link(&rest[2..end - 2], out))
        } else if rest.starts_with('[') && is_url(&rest[1..]) {
            rest.find(']').map(|end| {
                if let Some((_, label)) = rest[1..end].split_once(' ') {
                    strip_inline(label, out);
                }
                end + 1
            })
        } else if rest.starts_with('<') {
            html_tag(rest, out)
        } else if rest.starts_with("''") {
            Some(rest.len() - rest.trim_start_matches('\'').len())
        } else if rest.starts_with('&') {
            entity(rest, out)
        } else if rest.starts_with("__") {
            magic_word(rest)
        } else {
            None
        };
        if let Some(n) = consumed {
            i += n;
        } else {
            let c = rest.chars().next().expect("non-empty");
            out.push(c);
            i += c.len_utf8();
        }
    }
}

/// Byte length of `text` up to and including the `close` that matches the
/// `open` at its start, or `None` if it is never closed.
fn find_close(text: &str, open: &str, close: &str) -> Option<usize> {
    let mut depth = 0usize;
    let mut i = 0;
    while i < text.len() {
        let rest = &text[i..];
        if rest.starts_with(open) {
            depth += 1;
            i += open.len();
        } else if rest.starts_with(close) {
            depth -= 1;
            i += close.len();
            if depth == 0 {
                return Some(i);
            }
        } else {
            i += rest.chars().next().map_or(1, char::len_utf8);
        }
    }
    None
}

/// Splits on `|` outside nested templates and links.
fn split_params(text: &str) -> Vec<&str> {
    let mut params = Vec::new();
    let mut depth = 0usize;
    let mut start = 0;
    let mut i = 0;
    let bytes = text.as_bytes();
    while i < bytes.len() {
        match (bytes[i], bytes.get(i + 1)) {
            (b'{', Some(b'{')) | (b'[', Some(b'[')) => {
                depth += 1;
                i += 2;
            }
            (b'}', Some(b'}')) | (b']', Some(b']')) => {
                depth = depth.saturating_sub(1);
                i += 2;
            }
            (b'|', _) if depth == 0 => {
                params.push(&text[start..i]);
                i += 1;
                start = i;
            }
            _ => i += 1,
        }
    }
    params.push(&text[start..]);
    params
}

fn template(inner: &str, out: &mut String) {
    let params = split_params(inner);
    let name = params[0].trim().to_lowercase().replace('_', " ");
    let positional: Vec<&str> = params[1..]
        .iter()
        .copied()
        .filter(|p| !p.contains('='))
        .collect();
    let rendered = match name.as_str() {
        "lang" => positional.get(1).map(|text| vec![*text]),
        "convert" | "cvt" => positional.get(0..2).map(<[&str]>::to_vec),
        name if TEXT_TEMPLATES.contains(&name) => positional.first().map(|text| vec![*text]),
        _ => None,
    };
    if let Some(parts) = rendered {
        for (index, part) in parts.iter().enumerate() {
            if index > 0 {
                out.push(' ');
            }
            strip_inline(part.trim(), out);
        }
    }
}

fn link(inner: &str, out: &mut String) {
    let params = split_params(inner);
    let target = params[0].trim();
    if let Some((namespace, _)) = target.split_once(':') {
        let namespace = namespace.trim();
        // Interlanguage links use lowercase codes: [[fr:Chat]], [[simple:Cat]].
        let interlanguage = (2..=3).contains(&namespace.len())
            && namespace.bytes().all(|b| b.is_ascii_lowercase())
            || namespace == "simple";
        let dropped = DROPPED_NAMESPACES.contains(&namespace.to_lowercase().as_str());
        if !target.starts_with(':') && (dropped || interlanguage) {
            return;
        }
    }
    let target = target.trim_start_matches(':');
    match params.last() {
        Some(label) if params.len() > 1 && !label.trim().is_empty() => strip_inline(label, out),
        // The "pipe trick": [[Paris (France)|]] shows "Paris".
        Some(_) if params.len() > 1 => {
            let shown = target.split_once(" (").map_or(target, |(name, _)| name);
            out.push_str(shown);
        }
        _ => strip_inline(&target.replace('#', " "), out),
    }
}

fn is_url(text: &str) -> bool {
    ["http://", "https://", "//", "ftp://", "mailto:"]
        .iter()
        .any(|scheme| {
            text.len() >= scheme.len() && text[..scheme.len()].eq_ignore_ascii_case(scheme)
        })
}

/// Handles an HTML-like tag at the start of `rest`. Returns `None` if it is
/// not a tag, so the `<` is kept as text.
fn html_tag(rest: &str, out: &mut String) -> Option<usize> {
    let body = rest[1..].strip_prefix('/').unwrap_or(&rest[1..]);
    let closing = body.len() != rest.len() - 1;
    let name_len = body.bytes().take_while(u8::is_ascii_alphanumeric).count();
    if name_len == 0 || !body.as_bytes()[0].is_ascii_alphabetic() {
        return None;
    }
    let name = body[..name_len].to_ascii_lowercase();
    let after_name = &body[name_len..];
    if !after_name.starts_with(['>', '/', ' ', '\t', '\n']) {
        return None;
    }
    let tag_end = rest.find('>')? + 1;
    let self_closing = rest[..tag_end].ends_with("/>");

    if DROPPED_TAGS.contains(&name.as_str()) && !closing && !self_closing {
        let lower = rest.to_ascii_lowercase();
        let close = format!("</{name}");
        return Some(match lower[tag_end..].find(&close) {
            Some(offset) => {
                let close_start = tag_end + offset;
                rest[close_start..]
                    .find('>')
                    .map_or(rest.len(), |end| close_start + end + 1)
            }
            None => rest.len(),
        });
    }
    if matches!(name.as_str(), "br" | "p" | "div" | "li") {
        out.push('\n');
    }
    Some(tag_end)
}

/// Decodes an HTML entity such as `&nbsp;` or `&#8212;`.
fn entity(rest: &str, out: &mut String) -> Option<usize> {
    let end = rest[1..].find(';').filter(|&end| end <= 32)? + 1;
    let name = &rest[1..end];
    let decoded = if let Some(number) = name.strip_prefix('#') {
        let value = match number.strip_prefix(['x', 'X']) {
            Some(hex) => u32::from_str_radix(hex, 16).ok()?,
            None => number.parse().ok()?,
        };
        char::from_u32(value)?.to_string()
    } else {
        resolve_html5_entity(name)?.to_owned()
    };
    // Non-breaking and other special spaces become plain spaces.
    out.extend(
        decoded
            .chars()
            .map(|c| if c.is_whitespace() { ' ' } else { c }),
    );
    Some(end + 1)
}

/// Skips a behaviour switch such as `__NOTOC__`.
fn magic_word(rest: &str) -> Option<usize> {
    let name_len = rest[2..].bytes().take_while(u8::is_ascii_uppercase).count();
    (name_len > 0 && rest[2 + name_len..].starts_with("__")).then_some(name_len + 4)
}

/// Keeps prose lines: drops headings, lists, table leftovers and skipped
/// sections.
fn keep_prose_lines(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut skipping_below: Option<usize> = None;
    for line in text.lines() {
        let line = line.trim();
        if let Some(level) = heading_level(line) {
            let title = line.trim_matches('=').trim().to_lowercase();
            if skipping_below.is_some_and(|skip| level <= skip) {
                skipping_below = None;
            }
            if skipping_below.is_none() && SKIPPED_SECTIONS.contains(&title.as_str()) {
                skipping_below = Some(level);
            }
            out.push('\n');
            continue;
        }
        if skipping_below.is_some() {
            continue;
        }
        if line.is_empty()
            || line.starts_with(['*', '#', ':', ';', '|', '!', '{', '}'])
            || line.starts_with("----")
        {
            out.push('\n');
            continue;
        }
        out.push_str(line);
        out.push('\n');
    }
    out
}

fn heading_level(line: &str) -> Option<usize> {
    let level = line.bytes().take_while(|&b| b == b'=').count();
    (level >= 2 && line.len() > 2 * level && line.ends_with('=')).then_some(level)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plain(wikitext: &str) -> String {
        to_plain_text(wikitext)
            .lines()
            .filter(|line| !line.is_empty())
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn keeps_link_text() {
        assert_eq!(
            plain("A [[cat]] and [[Dog|dogs]] saw [[bus]]es in [[Paris (France)|]]."),
            "A cat and dogs saw buses in Paris."
        );
    }

    #[test]
    fn drops_files_categories_and_interlanguage_links() {
        assert_eq!(
            plain("[[File:A.jpg|thumb|A [[cat]] sleeping]]Text.[[Category:Cats]][[fr:Chat]]"),
            "Text."
        );
        assert_eq!(plain("See [[:Category:Cats|cats]]."), "See cats.");
    }

    #[test]
    fn drops_templates_but_renders_simple_ones() {
        assert_eq!(
            plain(
                "{{Infobox|name={{nowrap|X}}\n|size=1}}It is {{convert|5|km}} long, {{lang|fr|tr\u{e8}s}} {{nowrap|big}}.{{cite web|url=x}}"
            ),
            "It is 5 km long, tr\u{e8}s big."
        );
    }

    #[test]
    fn drops_refs_comments_tables_and_formatting() {
        let text = "'''Bold''' ''it''<ref name=\"a\">Cite</ref> x<ref name=b/><!-- c -->.\n{| class=wikitable\n| a || b\n|}\nAfter &amp; &nbsp;end&#46;";
        assert_eq!(plain(text), "Bold it x.\nAfter &  end.");
    }

    #[test]
    fn keeps_unknown_angle_brackets() {
        assert_eq!(
            plain("a < b and 3<4 <small>tiny</small>"),
            "a < b and 3<4 tiny"
        );
    }

    #[test]
    fn external_links() {
        assert_eq!(
            plain("[https://x.org Site name] and [http://y.org]."),
            "Site name and ."
        );
    }

    #[test]
    fn drops_headings_lists_and_reference_sections() {
        let text = "Intro.\n== History ==\nOld.\n* item\n== References ==\nRef text.\n=== Sub ===\nMore refs.\n== Life ==\nLived.\n__NOTOC__";
        assert_eq!(plain(text), "Intro.\nOld.\nLived.");
    }

    #[test]
    fn unclosed_markup_does_not_eat_text() {
        assert_eq!(
            plain("Some {{broken text [[also"),
            "Some {{broken text [[also"
        );
    }

    #[test]
    fn detects_disambiguation() {
        assert!(is_disambiguation("x\n{{Disambiguation}}"));
        assert!(!is_disambiguation("{{Infobox}}"));
    }
}
