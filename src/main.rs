use pyo3::prelude::*;
use pyo3::types::IntoPyDict;

fn main() -> PyResult<()> {
    Python::attach(|py| {
        // Import required modules
        let torch = py.import("torch")?;
        let torchaudio = py.import("torchaudio")?;
        let chatterbox_tts = py.import("chatterbox.tts")?;


        // INIT
        // -------------------------------------------------------------------------------------
        // Detect the best available device
        let device: &str = if torch.getattr("cuda")?.call_method0("is_available")?.extract()? {
            "cuda"
        } else if torch
            .getattr("backends")?
            .getattr("mps")?
            .call_method0("is_available")?
            .extract()?
        {
            "mps"
        } else {
            "cpu"
        };

        println!("\n-----------------------------------------------------------");
        println!("[DEBUG] Device & Environment Info");
        println!("-----------------------------------------------------------");
        println!("Using device: {}", device);

        let cuda_version = torch.getattr("version")?.getattr("cuda")?;
        println!("CUDA version: {}", cuda_version);

        let torch_version = torch.getattr("__version__")?;
        println!("PyTorch version: {}", torch_version);

        let cuda_available: bool = torch.getattr("cuda")?.call_method0("is_available")?.extract()?;
        println!("CUDA available: {}", cuda_available);

        if cuda_available {
            let cuda_device_count: i32 = torch.getattr("cuda")?.call_method0("device_count")?.extract()?;
            println!("CUDA device count: {}", cuda_device_count);
            let cuda_device_name = torch.getattr("cuda")?.call_method1("get_device_name", (0,))?;
            println!("CUDA device name: {}", cuda_device_name);
        }

        // INIT - END
        // -------------------------------------------------------------------------------------



        println!("\n-----------------------------------------------------------");
        println!("[DEBUG] Loading ChatterboxTTS Class");
        println!("-----------------------------------------------------------");
        // Load the model
        let chatterbox_class = chatterbox_tts.getattr("ChatterboxTTS")?;
        println!("[DEBUG] ChatterboxTTS class: {}", chatterbox_class.repr()?);
        println!("[DEBUG] ChatterboxTTS type: {}", chatterbox_class.get_type());

        println!("\n-----------------------------------------------------------");
        println!("[DEBUG] Creating kwargs");
        println!("-----------------------------------------------------------");
        println!("[DEBUG] Creating kwargs dict with device='{}'...", device);
        let kwargs = [("device", device)].into_py_dict(py)?;
        println!("[DEBUG] kwargs: {}", kwargs.repr()?);
        println!("Debugging: {:?}", kwargs.repr()?);





        println!("\n-----------------------------------------------------------");
        println!("[DEBUG] Loading Model from Pretrained");
        println!("-----------------------------------------------------------");
        println!("[DEBUG] Calling ChatterboxTTS.from_pretrained(device='{}')...", device);
        let model = chatterbox_class.call_method("from_pretrained", (), Some(&kwargs))?;
        println!("[DEBUG] Model loaded successfully!");
        println!("[DEBUG] Model type: {}", model.get_type());
        println!("[DEBUG] Model repr: {}", model.repr()?);





        println!("\n-----------------------------------------------------------");
        println!("[DEBUG] Generating Audio");
        println!("-----------------------------------------------------------");
        // Generate audio from text
        let text = "Ezreal and Jinx teamed up with Ahri, Yasuo, and Teemo to take down the enemy's Nexus in an epic late-game pentakill.";
        println!("[DEBUG] Generating audio for text: \"{}\"", text);
        println!("[DEBUG] Text length: {} characters", text.len());
        let wav = model.call_method1("generate", (text,))?;
        println!("[DEBUG] Audio generated successfully!");
        println!("[DEBUG] wav type: {}", wav.get_type());
        println!("[DEBUG] wav shape: {}", wav.getattr("shape")?);
        println!("[DEBUG] wav dtype: {}", wav.getattr("dtype")?);
        println!("[DEBUG] wav device: {}", wav.getattr("device")?);

        println!("\n-----------------------------------------------------------");
        println!("[DEBUG] Sample Rate Info");
        println!("-----------------------------------------------------------");
        // Get sample rate and save
        let sr = model.getattr("sr")?;
        println!("[DEBUG] Sample rate: {}", sr);
        println!("[DEBUG] Sample rate type: {}", sr.get_type());

        println!("\n-----------------------------------------------------------");
        println!("[DEBUG] Saving Audio File");
        println!("-----------------------------------------------------------");
        println!("[DEBUG] Saving audio to test-1.wav...");
        println!("[DEBUG] torchaudio.save('test-1.wav', wav, sr={})...", sr);
        torchaudio.call_method1("save", ("test-1.wav", wav, sr))?;

        println!("Audio saved to test-1.wav");
        Ok(())
    })
}
