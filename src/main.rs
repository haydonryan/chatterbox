use pyo3::prelude::*;
use pyo3::types::IntoPyDict;

fn main() -> PyResult<()> {
    Python::attach(|py| {
        // Import required modules
        let torch = py.import("torch")?;
        let torchaudio = py.import("torchaudio")?;
        let chatterbox_tts = py.import("chatterbox.tts")?;

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

        println!("Using device: {}", device);
        let cuda_version = torch.getattr("version")?.getattr("cuda")?;
        println!("CUDA version: {}", cuda_version);
        println!("\n-----------------------------------------------------------\n");

        // Load the model
        let chatterbox_class = chatterbox_tts.getattr("ChatterboxTTS")?;
        println!("\n-----------------------------------------------------------\n");
        let kwargs = [("device", device)].into_py_dict(py)?;
        println!("\n-----------------------------------------------------------\n");
        let model = chatterbox_class.call_method("from_pretrained", (), Some(&kwargs))?;

        println!("\n-----------------------------------------------------------\n");
        // Generate audio from text
        let text = "Ezreal and Jinx teamed up with Ahri, Yasuo, and Teemo to take down the enemy's Nexus in an epic late-game pentakill.";
        println!("\n-----------------------------------------------------------\n");
        let wav = model.call_method1("generate", (text,))?;

        println!("\n-----------------------------------------------------------\n");
        // Get sample rate and save
        let sr = model.getattr("sr")?;
        println!("\n-----------------------------------------------------------\n");
        println!("Saving....");
        torchaudio.call_method1("save", ("test-1.wav", wav, sr))?;

        println!("Audio saved to test-1.wav");
        Ok(())
    })
}
