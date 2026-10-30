//! Shared-format I/O/load-only progress regressions; no learned model execution.

#[cfg(test)]
mod gguf {
    use candle_gen::candle_core::quantized::gguf_file::*;
    use candle_gen::candle_core::Device;
    use candle_gen::gen_core::LoadActivityObserver;
    use std::sync::{Arc, Mutex};

    #[test]
    fn lazy_tensor_reads_report_unique_ranges_to_originating_request() {
        // A raw F32 container. Parsing/loading bytes performs no model or learned computation.
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&0x4655_4747u32.to_le_bytes());
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&1u64.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&1u64.to_le_bytes());
        bytes.push(b'w');
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&1u64.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        let header_len = bytes.len() as u64;
        bytes.resize(64, 0); // Alignment padding is not read and must not be charged.
        bytes.extend_from_slice(&0f32.to_le_bytes());
        let events = Arc::new(Mutex::new(Vec::new()));
        let output = events.clone();
        let mut reader = std::io::Cursor::new(bytes);
        let content = {
            let _observer = LoadActivityObserver::new(move |v| output.lock().unwrap().push(v));
            Content::read_with_path(&mut reader, "transformer/test.gguf").unwrap()
        };
        assert_eq!(events.lock().unwrap().last(), Some(&None));
        let _other = LoadActivityObserver::new(|_| panic!("lazy read changed request"));
        for _ in 0..2 {
            content.tensor(&mut reader, "w", &Device::Cpu).unwrap();
            assert_eq!(events.lock().unwrap().last(), Some(&None));
        }
        let events = events.lock().unwrap();
        let last = events.iter().flatten().next_back().unwrap();
        assert_eq!(last.bytes_read, header_len + 4);
        assert_eq!(last.total_bytes, Some(68));
        assert_eq!(last.component, "transformer");
    }

    #[test]
    fn malformed_header_clears_loading() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let output = events.clone();
        let _observer = LoadActivityObserver::new(move |v| output.lock().unwrap().push(v));
        assert!(
            Content::read_with_path(&mut std::io::Cursor::new(vec![0; 3]), "bad.gguf").is_err()
        );
        assert_eq!(events.lock().unwrap().last(), Some(&None));
        assert!(events
            .lock()
            .unwrap()
            .iter()
            .flatten()
            .any(|v| v.bytes_read == 3));
    }
}

#[cfg(test)]
mod ggml {
    use candle_gen::candle_core::quantized::ggml_file::*;
    use candle_gen::candle_core::Device;
    use candle_gen::gen_core::LoadActivityObserver;
    use std::sync::{Arc, Mutex};

    #[test]
    fn metadata_only_ggml_reports_reads_and_clears() {
        let mut bytes = 0x67676d6cu32.to_le_bytes().to_vec();
        bytes.extend_from_slice(&[0; 28]); // Empty vocab and no tensor records.
        let events = Arc::new(Mutex::new(Vec::new()));
        let output = events.clone();
        let _observer = LoadActivityObserver::new(move |v| output.lock().unwrap().push(v));
        Content::read_with_path(&mut std::io::Cursor::new(bytes), &Device::Cpu, "model.ggml")
            .unwrap();
        let events = events.lock().unwrap();
        let last = events.iter().flatten().next_back().unwrap();
        assert_eq!((last.bytes_read, last.total_bytes), (32, Some(32)));
        assert_eq!(events.last(), Some(&None));
    }
}

#[cfg(test)]
mod pth {
    use candle_gen::candle_core::pickle::*;
    use candle_gen::gen_core::LoadActivityObserver;
    use std::sync::{Arc, Mutex};

    #[test]
    fn pth_lazy_load_retains_request_and_deduplicates_ranges() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let output = events.clone();
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../vendor/candle-core/tests/fortran_tensor_3d.pth");
        let archive = {
            let _observer = LoadActivityObserver::new(move |v| output.lock().unwrap().push(v));
            PthTensors::new(&path, None).unwrap()
        };
        let _other = LoadActivityObserver::new(|_| panic!("lazy PTH changed request"));
        let name = archive.tensor_infos().keys().next().unwrap();
        archive.get(name).unwrap().unwrap();
        let first = events
            .lock()
            .unwrap()
            .iter()
            .flatten()
            .next_back()
            .unwrap()
            .bytes_read;
        archive.get(name).unwrap().unwrap();
        let events = events.lock().unwrap();
        let last = events.iter().flatten().next_back().unwrap();
        assert_eq!(last.bytes_read, first);
        assert!(last.bytes_read <= last.total_bytes.unwrap());
        assert_eq!(events.last(), Some(&None));
    }
}

#[cfg(test)]
mod numpy {
    use candle_gen::candle_core::npy::NpzTensors;
    use candle_gen::candle_core::Tensor;
    use candle_gen::gen_core::LoadActivityObserver;
    use std::path::Path;
    use std::sync::{Arc, Mutex};

    #[test]
    fn numpy_loads_and_lazy_archive_keep_source_accounting() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let output = events.clone();
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../vendor/candle-core/tests");
        let npy = root.join("test.npy");
        let npz = root.join("test.npz");
        let archive = {
            let _observer = LoadActivityObserver::new(move |v| output.lock().unwrap().push(v));
            Tensor::read_npy(&npy).unwrap();
            let last = events
                .lock()
                .unwrap()
                .iter()
                .flatten()
                .next_back()
                .cloned()
                .unwrap();
            assert_eq!(last.bytes_read, std::fs::metadata(&npy).unwrap().len());
            assert_eq!(events.lock().unwrap().last(), Some(&None));
            NpzTensors::new(&npz).unwrap()
        };
        let _other = LoadActivityObserver::new(|_| panic!("lazy NPZ changed request"));
        let name = archive.names()[0].clone();
        archive.get(&name).unwrap().unwrap();
        let first = events
            .lock()
            .unwrap()
            .iter()
            .flatten()
            .next_back()
            .unwrap()
            .bytes_read;
        archive.get(&name).unwrap().unwrap();
        let events = events.lock().unwrap();
        let last = events.iter().flatten().next_back().unwrap();
        assert_eq!(last.bytes_read, first);
        assert!(last.bytes_read <= last.total_bytes.unwrap());
        assert_eq!(events.last(), Some(&None));
    }
}
