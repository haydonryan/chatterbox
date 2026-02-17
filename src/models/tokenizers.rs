//! Rust wrapper for tokenization utilities.
//!
//! Mirrors the Python tokenizer implementations for English and multilingual text.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::cell::RefCell;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::error::{ChatterboxError, Result};

// Special tokens
pub const SOT: &str = "[START]";
pub const EOT: &str = "[STOP]";
pub const UNK: &str = "[UNK]";
pub const SPACE: &str = "[SPACE]";

// Model repository (for Cangjie mapping)
const REPO_ID: &str = "ResembleAI/chatterbox";

fn cast_dict<'py>(obj: &'py Bound<'py, PyAny>) -> Result<&'py Bound<'py, PyDict>> {
    obj.cast::<PyDict>()
        .map_err(|e| ChatterboxError::Python(pyo3::exceptions::PyTypeError::new_err(e.to_string())))
}

fn cast_list<'py>(obj: &'py Bound<'py, PyAny>) -> Result<&'py Bound<'py, PyList>> {
    obj.cast::<PyList>()
        .map_err(|e| ChatterboxError::Python(pyo3::exceptions::PyTypeError::new_err(e.to_string())))
}

fn nfkd_normalize_text(py: Python<'_>, text: &str) -> Result<String> {
    let unicodedata = py.import("unicodedata")?;
    let normalized: String = unicodedata.call_method1("normalize", ("NFKD", text))?.extract()?;
    Ok(normalized)
}

/// English tokenizer wrapper.
pub struct EnTokenizer {
    tokenizer: Py<PyAny>,
}

impl EnTokenizer {
    pub fn new(py: Python<'_>, vocab_file_path: &Path) -> Result<Self> {
        let tokenizers_mod = py.import("tokenizers")?;
        let tokenizer_class = tokenizers_mod.getattr("Tokenizer")?;
        let path_str = vocab_file_path.to_string_lossy();
        let tokenizer = tokenizer_class.call_method1("from_file", (path_str.as_ref(),))?;

        let instance = Self {
            tokenizer: tokenizer.into(),
        };
        instance.check_vocabset_sot_eot(py)?;
        Ok(instance)
    }

    pub fn as_py(&self) -> &Py<PyAny> {
        &self.tokenizer
    }

    fn check_vocabset_sot_eot(&self, py: Python<'_>) -> Result<()> {
        let vocab_any = self.tokenizer.bind(py).call_method0("get_vocab")?;
        let vocab = cast_dict(&vocab_any)?;
        if !vocab.contains(SOT)? || !vocab.contains(EOT)? {
            return Err(ChatterboxError::Python(
                pyo3::exceptions::PyAssertionError::new_err("Missing SOT/EOT in tokenizer vocab"),
            ));
        }
        Ok(())
    }

    pub fn encode(&self, py: Python<'_>, txt: &str) -> Result<Vec<i64>> {
        let txt = txt.replace(' ', SPACE);
        let encoding = self.tokenizer.bind(py).call_method1("encode", (txt,))?;
        let ids_any = encoding.getattr("ids")?;
        let ids: Vec<i64> = ids_any.extract()?;
        Ok(ids)
    }

    pub fn text_to_tokens(&self, py: Python<'_>, text: &str) -> Result<Py<PyAny>> {
        let ids = self.encode(py, text)?;
        let torch = py.import("torch")?;
        let tensor = torch.call_method1("IntTensor", (ids,))?;
        let tensor = tensor.call_method1("unsqueeze", (0,))?;
        Ok(tensor.into())
    }

    pub fn decode(&self, py: Python<'_>, seq: &Bound<'_, PyAny>) -> Result<String> {
        let torch = py.import("torch")?;
        let is_tensor: bool = torch
            .getattr("is_tensor")?
            .call1((seq,))?
            .extract()?;

        let seq_any = if is_tensor {
            seq.call_method0("cpu")?.call_method0("numpy")?
        } else {
            seq.clone()
        };

        let kwargs = PyDict::new(py);
        kwargs.set_item("skip_special_tokens", false)?;
        let decoded_any = self
            .tokenizer
            .bind(py)
            .call_method("decode", (seq_any,), Some(&kwargs))?;
        let mut txt: String = decoded_any.extract()?;

        txt = txt.replace(' ', "");
        txt = txt.replace(SPACE, " ");
        txt = txt.replace(EOT, "");
        txt = txt.replace(UNK, "");
        Ok(txt)
    }
}

fn is_kanji(c: char) -> bool {
    let code = c as u32;
    (19968..=40959).contains(&code)
}

fn is_katakana(c: char) -> bool {
    let code = c as u32;
    (12449..=12538).contains(&code)
}

fn korean_normalize(text: &str) -> String {
    fn decompose_hangul(ch: char) -> String {
        let code = ch as u32;
        if !(0xAC00..=0xD7AF).contains(&code) {
            return ch.to_string();
        }

        let base = code - 0xAC00;
        let initial = 0x1100 + base / (21 * 28);
        let medial = 0x1161 + (base % (21 * 28)) / 28;
        let final_jamo = base % 28;

        let mut out = String::new();
        out.push(char::from_u32(initial).unwrap());
        out.push(char::from_u32(medial).unwrap());
        if final_jamo > 0 {
            out.push(char::from_u32(0x11A7 + final_jamo).unwrap());
        }
        out
    }

    let mut out = String::new();
    for ch in text.chars() {
        out.push_str(&decompose_hangul(ch));
    }
    out.trim().to_string()
}

struct ChineseCangjieConverter {
    word2cj: HashMap<String, String>,
    cj2word: HashMap<String, Vec<String>>,
    segmenter: Option<Py<PyAny>>,
}

impl ChineseCangjieConverter {
    fn new(py: Python<'_>, model_dir: Option<&Path>) -> Result<Self> {
        let mut converter = Self {
            word2cj: HashMap::new(),
            cj2word: HashMap::new(),
            segmenter: None,
        };
        converter.load_cangjie_mapping(py, model_dir)?;
        converter.init_segmenter(py);
        Ok(converter)
    }

    fn load_cangjie_mapping(&mut self, py: Python<'_>, model_dir: Option<&Path>) -> Result<()> {
        let hf_mod = py.import("huggingface_hub")?;
        let hf_download = hf_mod.getattr("hf_hub_download")?;
        let kwargs = PyDict::new(py);
        if let Some(dir) = model_dir {
            let dir_str = dir.to_string_lossy();
            kwargs.set_item("cache_dir", dir_str.as_ref())?;
        }
        let cangjie_path = hf_download.call((REPO_ID, "Cangjie5_TC.json"), Some(&kwargs))?;
        let cangjie_path: String = cangjie_path.extract()?;

        let builtins = py.import("builtins")?;
        let open = builtins.getattr("open")?;
        let file = open.call1((cangjie_path.as_str(), "r"))?;
        let json_mod = py.import("json")?;
        let data = json_mod.call_method1("load", (file.clone(),))?;
        let _ = file.call_method0("close");

        let data_list = cast_list(&data)?;
        for entry in data_list.iter() {
            let entry: String = entry.extract()?;
            let mut parts = entry.split('\t');
            let word = match parts.next() {
                Some(w) => w,
                None => continue,
            };
            let code = match parts.next() {
                Some(c) => c,
                None => continue,
            };
            self.word2cj.insert(word.to_string(), code.to_string());
            self.cj2word
                .entry(code.to_string())
                .or_insert_with(Vec::new)
                .push(word.to_string());
        }
        Ok(())
    }

    fn init_segmenter(&mut self, py: Python<'_>) {
        match py.import("spacy_pkuseg") {
            Ok(mod_) => {
                if let Ok(pkuseg) = mod_.getattr("pkuseg") {
                    if let Ok(segmenter) = pkuseg.call0() {
                        self.segmenter = Some(segmenter.into());
                        return;
                    }
                }
                eprintln!("pkuseg not available - Chinese segmentation will be skipped");
            }
            Err(_) => {
                eprintln!("pkuseg not available - Chinese segmentation will be skipped");
            }
        }
        self.segmenter = None;
    }

    fn cangjie_encode(&self, glyph: &str) -> Option<String> {
        let code = self.word2cj.get(glyph)?;
        let words = self.cj2word.get(code)?;
        let index = words.iter().position(|w| w == glyph)?;
        let suffix = if index > 0 { index.to_string() } else { String::new() };
        Some(format!("{}{}", code, suffix))
    }

    fn convert(&self, py: Python<'_>, text: &str) -> Result<String> {
        let full_text = if let Some(ref segmenter) = self.segmenter {
            let segmented = segmenter.bind(py).call_method1("cut", (text,))?;
            let seg_list = cast_list(&segmented)?;
            let mut parts = Vec::with_capacity(seg_list.len());
            for item in seg_list.iter() {
                let word: String = item.extract()?;
                parts.push(word);
            }
            parts.join(" ")
        } else {
            text.to_string()
        };

        let unicodedata = py.import("unicodedata")?;
        let category_fn = unicodedata.getattr("category")?;

        let mut output = String::new();
        for ch in full_text.chars() {
            let cat: String = category_fn.call1((ch.to_string(),))?.extract()?;
            if cat == "Lo" {
                if let Some(code) = self.cangjie_encode(&ch.to_string()) {
                    for c in code.chars() {
                        output.push_str("[cj_");
                        output.push(c);
                        output.push(']');
                    }
                    output.push_str("[cj_.]");
                } else {
                    output.push(ch);
                }
            } else {
                output.push(ch);
            }
        }
        Ok(output)
    }
}

/// Multilingual tokenizer wrapper.
pub struct MTLTokenizer {
    tokenizer: Py<PyAny>,
    cangjie: ChineseCangjieConverter,
    kakasi: RefCell<Option<Py<PyAny>>>,
    dicta: RefCell<Option<Py<PyAny>>>,
    russian_stresser: RefCell<Option<Py<PyAny>>>,
}

impl MTLTokenizer {
    pub fn new(py: Python<'_>, vocab_file_path: &Path) -> Result<Self> {
        let tokenizers_mod = py.import("tokenizers")?;
        let tokenizer_class = tokenizers_mod.getattr("Tokenizer")?;
        let path_str = vocab_file_path.to_string_lossy();
        let tokenizer = tokenizer_class.call_method1("from_file", (path_str.as_ref(),))?;
        let model_dir = vocab_file_path.parent().map(PathBuf::from);
        let cangjie = ChineseCangjieConverter::new(py, model_dir.as_deref())?;

        let instance = Self {
            tokenizer: tokenizer.into(),
            cangjie,
            kakasi: RefCell::new(None),
            dicta: RefCell::new(None),
            russian_stresser: RefCell::new(None),
        };
        instance.check_vocabset_sot_eot(py)?;
        Ok(instance)
    }

    pub fn as_py(&self) -> &Py<PyAny> {
        &self.tokenizer
    }

    fn check_vocabset_sot_eot(&self, py: Python<'_>) -> Result<()> {
        let vocab_any = self.tokenizer.bind(py).call_method0("get_vocab")?;
        let vocab = cast_dict(&vocab_any)?;
        if !vocab.contains(SOT)? || !vocab.contains(EOT)? {
            return Err(ChatterboxError::Python(
                pyo3::exceptions::PyAssertionError::new_err("Missing SOT/EOT in tokenizer vocab"),
            ));
        }
        Ok(())
    }

    pub fn preprocess_text(
        &self,
        py: Python<'_>,
        raw_text: &str,
        lowercase: bool,
        nfkd_normalize: bool,
    ) -> Result<String> {
        let mut text = raw_text.to_string();
        if lowercase {
            text = text.to_lowercase();
        }
        if nfkd_normalize {
            text = nfkd_normalize_text(py, &text)?;
        }
        Ok(text)
    }

    fn get_kakasi(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        if let Some(kakasi) = self.kakasi.borrow().as_ref() {
            return Some(kakasi.clone_ref(py));
        }
        match py.import("pykakasi") {
            Ok(mod_) => match mod_.getattr("kakasi").and_then(|k| k.call0()) {
                Ok(kakasi) => {
                    let kakasi: Py<PyAny> = kakasi.into();
                    *self.kakasi.borrow_mut() = Some(kakasi.clone_ref(py));
                    Some(kakasi)
                }
                Err(_) => {
                    eprintln!("pykakasi not available - Japanese text processing skipped");
                    None
                }
            },
            Err(_) => {
                eprintln!("pykakasi not available - Japanese text processing skipped");
                None
            }
        }
    }

    fn hiragana_normalize(&self, py: Python<'_>, text: &str) -> Result<String> {
        let kakasi = match self.get_kakasi(py) {
            Some(k) => k,
            None => return Ok(text.to_string()),
        };

        let result = kakasi.bind(py).call_method1("convert", (text,))?;
        let result_list = cast_list(&result)?;
        let mut out = String::new();

        for item in result_list.iter() {
            let item = cast_dict(&item)?;
            let inp_item = item
                .get_item("orig")?
                .ok_or_else(|| ChatterboxError::Python(pyo3::exceptions::PyKeyError::new_err("orig")))?;
            let hira_item = item
                .get_item("hira")?
                .ok_or_else(|| ChatterboxError::Python(pyo3::exceptions::PyKeyError::new_err("hira")))?;
            let inp: String = inp_item.extract()?;
            let mut hira: String = hira_item.extract()?;

            if inp.chars().any(is_kanji) {
                if !hira.is_empty() {
                    let first = hira.chars().next().unwrap();
                    if first == '\u{306F}' || first == '\u{3078}' {
                        hira = format!(" {}", hira);
                    }
                }
                out.push_str(&hira);
            } else if !inp.is_empty() && inp.chars().all(is_katakana) {
                out.push_str(&inp);
            } else {
                out.push_str(&inp);
            }
        }

        let normalized = nfkd_normalize_text(py, &out)?;
        Ok(normalized)
    }

    fn get_dicta(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        if let Some(dicta) = self.dicta.borrow().as_ref() {
            return Some(dicta.clone_ref(py));
        }
        match py.import("dicta_onnx") {
            Ok(mod_) => match mod_.getattr("Dicta").and_then(|d| d.call0()) {
                Ok(dicta) => {
                    let dicta: Py<PyAny> = dicta.into();
                    *self.dicta.borrow_mut() = Some(dicta.clone_ref(py));
                    Some(dicta)
                }
                Err(_) => {
                    eprintln!("dicta_onnx not available - Hebrew text processing skipped");
                    None
                }
            },
            Err(_) => {
                eprintln!("dicta_onnx not available - Hebrew text processing skipped");
                None
            }
        }
    }

    fn add_hebrew_diacritics(&self, py: Python<'_>, text: &str) -> Result<String> {
        let dicta = match self.get_dicta(py) {
            Some(d) => d,
            None => return Ok(text.to_string()),
        };

        match dicta.bind(py).call_method1("add_diacritics", (text,)) {
            Ok(result) => Ok(result.extract()?),
            Err(err) => {
                eprintln!("Hebrew diacritization failed: {}", err);
                Ok(text.to_string())
            }
        }
    }

    fn get_russian_stresser(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        if let Some(stresser) = self.russian_stresser.borrow().as_ref() {
            return Some(stresser.clone_ref(py));
        }
        match py.import("russian_text_stresser.text_stresser") {
            Ok(mod_) => match mod_.getattr("RussianTextStresser").and_then(|s| s.call0()) {
                Ok(stresser) => {
                    let stresser: Py<PyAny> = stresser.into();
                    *self.russian_stresser.borrow_mut() = Some(stresser.clone_ref(py));
                    Some(stresser)
                }
                Err(_) => {
                    eprintln!("russian_text_stresser not available - Russian stress labeling skipped");
                    None
                }
            },
            Err(_) => {
                eprintln!("russian_text_stresser not available - Russian stress labeling skipped");
                None
            }
        }
    }

    fn add_russian_stress(&self, py: Python<'_>, text: &str) -> Result<String> {
        let stresser = match self.get_russian_stresser(py) {
            Some(s) => s,
            None => return Ok(text.to_string()),
        };

        match stresser.bind(py).call_method1("stress_text", (text,)) {
            Ok(result) => Ok(result.extract()?),
            Err(err) => {
                eprintln!("Russian stress labeling failed: {}", err);
                Ok(text.to_string())
            }
        }
    }

    pub fn encode(
        &self,
        py: Python<'_>,
        txt: &str,
        language_id: Option<&str>,
        lowercase: bool,
        nfkd_normalize: bool,
    ) -> Result<Vec<i64>> {
        let mut txt = self.preprocess_text(py, txt, lowercase, nfkd_normalize)?;

        if let Some(lang) = language_id {
            match lang {
                "zh" => {
                    txt = self.cangjie.convert(py, &txt)?;
                }
                "ja" => {
                    txt = self.hiragana_normalize(py, &txt)?;
                }
                "he" => {
                    txt = self.add_hebrew_diacritics(py, &txt)?;
                }
                "ko" => {
                    txt = korean_normalize(&txt);
                }
                "ru" => {
                    txt = self.add_russian_stress(py, &txt)?;
                }
                _ => {}
            }

            txt = format!("[{}]{}", lang.to_lowercase(), txt);
        }

        txt = txt.replace(' ', SPACE);
        let encoding = self.tokenizer.bind(py).call_method1("encode", (txt,))?;
        let ids_any = encoding.getattr("ids")?;
        let ids: Vec<i64> = ids_any.extract()?;
        Ok(ids)
    }

    pub fn text_to_tokens(
        &self,
        py: Python<'_>,
        text: &str,
        language_id: Option<&str>,
        lowercase: bool,
        nfkd_normalize: bool,
    ) -> Result<Py<PyAny>> {
        let ids = self.encode(py, text, language_id, lowercase, nfkd_normalize)?;
        let torch = py.import("torch")?;
        let tensor = torch.call_method1("IntTensor", (ids,))?;
        let tensor = tensor.call_method1("unsqueeze", (0,))?;
        Ok(tensor.into())
    }

    pub fn decode(&self, py: Python<'_>, seq: &Bound<'_, PyAny>) -> Result<String> {
        let torch = py.import("torch")?;
        let is_tensor: bool = torch
            .getattr("is_tensor")?
            .call1((seq,))?
            .extract()?;

        let seq_any = if is_tensor {
            seq.call_method0("cpu")?.call_method0("numpy")?
        } else {
            seq.clone()
        };

        let kwargs = PyDict::new(py);
        kwargs.set_item("skip_special_tokens", false)?;
        let decoded_any = self
            .tokenizer
            .bind(py)
            .call_method("decode", (seq_any,), Some(&kwargs))?;
        let mut txt: String = decoded_any.extract()?;

        txt = txt.replace(' ', "");
        txt = txt.replace(SPACE, " ");
        txt = txt.replace(EOT, "");
        txt = txt.replace(UNK, "");
        Ok(txt)
    }
}
