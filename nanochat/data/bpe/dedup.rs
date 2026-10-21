//! Deduplication of curation (nanochat/data/dedup.py): each document's MinHash signature (sketch), and the documents
//! that are exact or near duplicates of an earlier one (duplicates).
//!
//! A document is the set of its n-grams of words as StarCoder2 (arXiv 2402.19173) near-deduplicates code
//! (bigcode-dataset near_deduplication/bigcode-v2/intra_dedup.py `ngrams`, min_length = n): the text lowercased and
//! split at every non-word character (`re.split(r"\W", content.lower())`, so a run of k non-word characters leaves
//! k - 1 empty words; a word character is alphanumeric or '_', Rust's char::is_alphanumeric, which unlike Python's `\w`
//! also counts the Unicode Other_Alphabetic marks); none below n words. A word is hashed by rustc-hash's FxHasher
//! (rensa's token hash), an n-gram by folding its words' hashes through splitmix64.
//!
//! The signature is rensa's R-MinHash (github.com/beowolx/rensa src/rminhash/bucket.rs, after Fast Similarity
//! Sketching, Dahlgaard et al. arXiv 1704.04370): in each of its rounds every n-gram is placed in one slot by Lemire's
//! multiply-shift (arXiv 1805.10941) of the low 32 bits of a splitmix64 mix (without rensa's rejection step for a slot
//! count that is not a power of two), its rank the round then the mix's top 30 bits; a slot that no round fills takes
//! the minimum over the n-grams of its own mix, ranked after every round. Each slot's minimum is that of one fixed
//! random function of (n-gram, slot), so two documents' slots agree with probability their Jaccard similarity. A slot
//! is stored as the low byte of its minimum (b-bit minwise hashing, Li and König, WWW 2010, arXiv 0910.3349).
//!
//! Duplicates: documents with the same text are one cluster (exact before near deduplication, as The Stack
//! (arXiv 2211.15533) and StarCoder2); documents with the same slots in any LSH band are candidates, joined when their
//! Jaccard similarity estimated from all slots reaches the threshold (The Stack v3 verifies LSH candidates by estimated
//! similarity; StarCoder2 joins every candidate). Within a band's bucket each document, in document order, joins the
//! first similar one of the bucket's leaders (the first document of each cluster met in it), else leads. Union-find
//! (petgraph) joins clusters across bands; each cluster keeps its first document (PUFFER's keep-first order, Zyphra
//! arXiv 2608.28622). A document without an n-gram is in no band (StarCoder2's all-maximum signature would put every
//! such document in one bucket), so only its exact copies join it.

use std::fs::File;
use std::hash::Hasher;

use memmap2::Mmap;
use petgraph::unionfind::UnionFind;
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedBytes;
use pyo3::types::PyBytes;
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use rustc_hash::FxHasher;

/// rensa's R-MinHash: three bucket rounds; a rank holds its round (or the fallback after the rounds) in its top bits
/// and the mix's top bits below.
const ROUNDS: u32 = 3;
const FRACTION_BITS: u32 = u32::BITS - (ROUNDS + 1).ilog2();
const FALLBACK: u32 = ROUNDS << FRACTION_BITS;

/// SplitMix64's output function (Steele, Lea and Flood, OOPSLA 2014, "Fast Splittable Pseudorandom Number Generators").
fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

/// Calls visit with the hash of each n-gram of a lowercased text, streaming through a ring of the last n words'
/// hashes, so that a document needs no memory beyond its lowercased copy.
fn for_each_ngram(lower: &str, n: usize, mut visit: impl FnMut(u64)) {
    let mut ring = vec![0u64; n];
    for (i, word) in lower.split(|c: char| !(c.is_alphanumeric() || c == '_')).enumerate() {
        let mut hasher = FxHasher::default();
        hasher.write(word.as_bytes());
        ring[i % n] = hasher.finish();
        if i + 1 >= n {
            visit((0..n).fold(0, |h, j| splitmix64(h ^ ring[(i + 1 + j) % n])));
        }
    }
}

/// The signatures of texts (UTF-8 bytes): slots bytes per document (zeros without an n-gram), and per document whether
/// it has an n-gram (1) or not (0). The seeds (one per slot for the fallback, then one per round) are ChaCha8's draws
/// under the zero key.
#[pyfunction]
#[pyo3(signature = (texts, *, n, slots))]
pub(crate) fn sketch<'py>(py: Python<'py>, texts: Vec<PyBackedBytes>, n: usize, slots: usize) -> (Bound<'py, PyBytes>, Bound<'py, PyBytes>) {
    let mut rng = ChaCha8Rng::from_seed(Default::default());
    let seeds: Vec<u64> = (0..slots + ROUNDS as usize).map(|_| rng.random()).collect();
    let (signatures, shingled) = py.detach(|| {
        let (mut signatures, mut shingled) = (vec![0u8; texts.len() * slots], vec![0u8; texts.len()]);
        let (mut values, mut unfilled) = (vec![0u32; slots], Vec::with_capacity(slots));
        for (d, text) in texts.iter().enumerate() {
            let lower = std::str::from_utf8(text).unwrap().to_lowercase();
            values.fill(u32::MAX);
            for_each_ngram(&lower, n, |h| {
                shingled[d] = 1;
                for round in 0..ROUNDS {
                    let mixed = splitmix64(h ^ seeds[slots + round as usize]);
                    let slot = ((mixed as u32 as u64 * slots as u64) >> u32::BITS) as usize;
                    values[slot] = values[slot].min((round << FRACTION_BITS) | (mixed >> (u64::BITS - FRACTION_BITS)) as u32);
                }
            });
            if shingled[d] == 0 {
                continue; // no n-gram: its signature stays zeros
            }
            // a round's rank is below FALLBACK, so a slot still at u32::MAX was filled by no round
            unfilled.clear();
            unfilled.extend((0..slots).filter(|&j| values[j] == u32::MAX));
            if !unfilled.is_empty() {
                for_each_ngram(&lower, n, |h| {
                    for &j in &unfilled {
                        values[j] = values[j].min(FALLBACK | (splitmix64(h ^ seeds[j]) >> (u64::BITS - FRACTION_BITS)) as u32);
                    }
                });
            }
            for (byte, value) in signatures[d * slots..(d + 1) * slots].iter_mut().zip(&values) {
                *byte = *value as u8;
            }
        }
        (signatures, shingled)
    });
    (PyBytes::new(py, &signatures), PyBytes::new(py, &shingled))
}

/// The documents d < n that are members, by key (in parallel), each as (key, d).
fn sorted<'a>(n: usize, key: impl Fn(u32) -> &'a [u8] + Sync, member: impl Fn(usize) -> bool + Sync) -> Vec<(&'a [u8], u32)> {
    let mut keys: Vec<(&[u8], u32)> = (0..n).into_par_iter().filter(|&d| member(d)).map(|d| (key(d as u32), d as u32)).collect();
    keys.par_sort_unstable();
    keys
}

/// Which documents a deduplication keeps (1) or drops (0), from the files of its documents in order: their signatures
/// (sketch), ids (equal for equal texts), whether each takes part and whether it has an n-gram (sketch). A document
/// that takes part is kept when it is the first of its cluster, one that does not is dropped; bands of rows slots each
/// (slots / rows bands); threshold the Jaccard similarity of a near duplicate.
#[pyfunction]
#[pyo3(signature = (signatures, ids, takes_part, shingled, *, rows, threshold))]
pub(crate) fn duplicates<'py>(py: Python<'py>, signatures: &str, ids: &str, takes_part: &str, shingled: &str, rows: usize, threshold: f64) -> Bound<'py, PyBytes> {
    let map = |path: &str| unsafe { Mmap::map(&File::open(path).unwrap()).unwrap() }; // written once by dedup.py, never modified
    let (signatures, ids, takes_part, shingled) = (map(signatures), map(ids), map(takes_part), map(shingled));
    let keep = py.detach(|| {
        let n = takes_part.len();
        let (slots, width) = (signatures.len() / n, ids.len() / n);
        let signature = |d: u32| &signatures[d as usize * slots..(d as usize + 1) * slots];
        // least: the slots two documents of Jaccard similarity threshold share in expectation, a byte of two different
        // minima agreeing with probability 1 / 2^8 (Li and König); at least least equal slots = estimated J >= threshold
        let chance = 1.0 / f64::from(1u32 << u8::BITS);
        let least = (slots as f64 * (threshold + (1.0 - threshold) * chance)).ceil() as usize;
        let similar = |x: u32, y: u32| signature(x).iter().zip(signature(y)).filter(|(a, b)| a == b).count() >= least;
        let mut clusters = UnionFind::<u32>::new(n);
        let id = |d: u32| &ids[d as usize * width..(d as usize + 1) * width];
        for pair in sorted(n, id, |d| takes_part[d] == 1).windows(2).filter(|pair| pair[0].0 == pair[1].0) {
            clusters.union(pair[0].1, pair[1].1);
        }
        for band in 0..slots / rows {
            let rows_of = |d: u32| &signature(d)[band * rows..(band + 1) * rows];
            for bucket in sorted(n, rows_of, |d| takes_part[d] == 1 && shingled[d] == 1).chunk_by(|a, b| a.0 == b.0).filter(|bucket| bucket.len() > 1) {
                let mut leaders: Vec<u32> = Vec::new();
                for &(_, d) in bucket {
                    let root = clusters.find_mut(d);
                    if leaders.iter().any(|&l| clusters.find_mut(l) == root) {
                        continue;
                    }
                    if let Some(l) = leaders.iter().copied().find(|&l| similar(l, d)) {
                        clusters.union(l, d);
                    } else {
                        leaders.push(d);
                    }
                }
            }
        }
        let labels = clusters.into_labeling();
        let mut first = vec![u32::MAX; n];
        for (d, &root) in labels.iter().enumerate() {
            first[root as usize] = first[root as usize].min(d as u32);
        }
        (0..n).map(|d| u8::from(takes_part[d] == 1 && first[labels[d] as usize] == d as u32)).collect::<Vec<u8>>()
    });
    PyBytes::new(py, &keep)
}
