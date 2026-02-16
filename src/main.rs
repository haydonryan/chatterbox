mod chatterbox;
mod error;

use pyo3::prelude::*;
use std::path::Path;

use chatterbox::{print_device_info, ChatterboxTTS, Device, GenerateOptions};
use error::Result;

fn main() -> Result<()> {
    Python::attach(|py| {
        // Print environment info
        println!("-----------------------------------------------------------");
        println!("Device & Environment Info");
        println!("-----------------------------------------------------------");
        print_device_info(py)?;

        // Detect best device
        let device = Device::detect(py)?;
        println!("\nUsing device: {}", device);

        // Load model
        println!("\n-----------------------------------------------------------");
        println!("Loading ChatterboxTTS Model");
        println!("-----------------------------------------------------------");
        let tts = ChatterboxTTS::from_pretrained(py, device)?;
        println!("Model loaded successfully!");
        println!("Sample rate: {} Hz", tts.sample_rate(py)?);

        // Generate audio
        println!("\n-----------------------------------------------------------");
        println!("Generating Audio");
        println!("-----------------------------------------------------------");
        let text = "Hello there, this is a test of chatterbox text to speech";
                   // #to take down the enemy's Nexus in an epic late-game pentakill.";
        //let text = "Ezreal and Jinx teamed up with Ahri, Yasuo, and Teemo \
        //            to take down the enemy's Nexus in an epic late-game pentakill.";
        println!("Text: \"{}\"", text);

        let options = GenerateOptions::new().temperature(0.8).cfg_weight(0.5);

        let audio = tts.generate(py, text, options)?;

        let (channels, samples) = audio.shape(py)?;
        println!(
            "Generated audio: {} channels, {} samples",
            channels, samples
        );
        println!(
            "Duration: {:.2}s",
            samples as f32 / audio.sample_rate() as f32
        );

        // Save output
        println!("\n-----------------------------------------------------------");
        println!("Saving Audio");
        println!("-----------------------------------------------------------");
        let output_path = Path::new("test-1.wav");
        audio.save_wav(py, output_path)?;
        println!("Audio saved to {}", output_path.display());

        Ok(())
    })
}
