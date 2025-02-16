// main.rs

use core::time::Duration;
use rodio::{OutputStream, source::Source};
use std::collections::HashMap;
use std::error::Error;
use std::io::{stdin, stdout, Write};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread;
use midir::Ignore;
use midir::MidiInput;

#[derive(Debug)]
struct NoteData {
    note: std::thread::JoinHandle<()>,
    shared: Arc<Mutex<f32>>
}

impl NoteData {
    fn new(note: std::thread::JoinHandle<()>, shared: Arc<Mutex<f32>>) -> NoteData {
        return NoteData {
            note: note,
            shared: shared,
        }
    }
}

struct WavetableOscillator {
    sample_rate: u32,
    wave_table: Vec<f32>,
    index: f32,
    index_increment: f32,
    amplitude: f32,
}

impl WavetableOscillator {
    fn new(sample_rate: u32, wave_table: Vec<f32>) -> WavetableOscillator {
        return WavetableOscillator {
            sample_rate: sample_rate,
            wave_table: wave_table,
            index: 0.0,
            index_increment: 0.0,
            amplitude: 1.0,
        }
    }

    fn set_frequency(&mut self, frequency: f32) {
        self.index_increment = frequency * self.wave_table.len() as f32 / self.sample_rate as f32;
    }

    fn set_amplitude(&mut self, amplitude: f32) {
        self.amplitude = amplitude;
    }

    fn get_sample(&mut self) -> f32 {
        let sample = self.lerp_sine();
        self.index += self.index_increment;
        self.index %= self.wave_table.len() as f32;
        return sample;
    }

    fn lerp_saw(&self) -> f32 {
        let truncated_index = self.index as usize;
        let next_index = (truncated_index + 1) % self.wave_table.len();

        let next_index_weight = self.index - truncated_index as f32;
        let truncated_index_weight = 1.0 - next_index_weight;

        return truncated_index_weight * self.wave_table[truncated_index] + next_index_weight * self.wave_table[next_index];
    }

    fn lerp_sine(&self) -> f32 {
        let truncated_index = self.index as usize;
        let next_index = (truncated_index + 1) % self.wave_table.len();

        let next_index_weight = self.index - truncated_index as f32;
        let truncated_index_weight = 1.0 - next_index_weight;

        return self.amplitude * (truncated_index_weight * self.wave_table[truncated_index] + next_index_weight * self.wave_table[next_index]);
    }

    fn lerp_square(&self) -> f32 {
        let truncated_index = self.index as usize;
        let next_index = (truncated_index + 1) % self.wave_table.len();

        let next_index_weight = self.index - truncated_index as f32;
        let truncated_index_weight = 1.0 - next_index_weight;

        //return truncated_index_weight * self.wave_table[truncated_index] + next_index_weight * self.wave_table[next_index];
        let ret: f32 = truncated_index_weight * self.wave_table[truncated_index] + next_index_weight * self.wave_table[next_index];
        if ret > 0.0 {
            return 1.0;
        }
        return -1.0;
    }
}

impl Iterator for WavetableOscillator {
    type Item = f32;

    fn next(&mut self) -> Option<Self::Item> {
        return Some(self.get_sample());
    }
}

impl Source for WavetableOscillator {
    fn channels(&self) -> u16 {
        return 1;
    }

    fn sample_rate(&self) -> u32 {
        return self.sample_rate;
    }

    fn current_frame_len(&self) -> Option<usize> {
        return None;
    }

    fn total_duration(&self) -> Option<Duration> {
        return None;
    }
}

//fn wavetable_main(frequency: f32, velocity: f32, tx: std::sync::mpsc::Sender<()>, rx: std::sync::mpsc::Receiver<()>) -> thread::JoinHandle<()> {
fn wavetable_main(frequency: f32, velocity: f32, shared: Arc<Mutex<f32>>) -> thread::JoinHandle<()> {
    let note = std::thread::spawn(move ||  {
        let wave_table_size = 64;
        let mut wave_table: Vec<f32> = Vec::with_capacity(wave_table_size);
        for n in 0..wave_table_size {
            wave_table.push((2.0 * std::f32::consts::PI * n as f32 / wave_table_size as f32).sin());
        }
        let mut oscillator = WavetableOscillator::new(44100, wave_table);
        oscillator.set_frequency(frequency);
        oscillator.set_amplitude(velocity/127.0);

        let (_stream, stream_handle) = OutputStream::try_default().unwrap();
        let _result = stream_handle.play_raw(oscillator.convert_samples());
        //std::thread::sleep(std::time::Duration::from_millis(200));
        let mut run_loop: bool = true;
        while run_loop {
            let shared_vel = shared.lock().unwrap();
            println!("Shared! {}", *shared_vel);
            run_loop = *shared_vel > 0.0;
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    });
    return note;
}

fn main() {
    match run() {
        Ok(_) => (),
        Err(err) => println!("Error: {}", err),
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let mut input = String::new();
    //let mut notes: Vec<std::thread::JoinHandle<()>> = Vec::with_capacity(16);
    //let mut notes: HashMap<u8, (thread::JoinHandle<()>, Arc<f32>)> = HashMap::with_capacity(16);
    let mut notes: HashMap<u8, NoteData> = HashMap::with_capacity(16);
    let voices: [usize; 16] = core::array::from_fn(|i| i+1);

    let mut midi_in= MidiInput::new("midi_read_fx")?;
    midi_in.ignore(Ignore::None);

    // Get an input port (read from console if multiple are available)
    let in_ports = midi_in.ports();
    let in_port = match in_ports.len() {
        0 => return Err("no input port found".into()),
        1 => {
            println!(
                "Choosing the only available input port: {}",
                midi_in.port_name(&in_ports[0]).unwrap()
            );
            &in_ports[0]
        }
        _ => {
            println!("\nAvailable input ports:");
            for (i, p) in in_ports.iter().enumerate() {
                println!("{}: {}", i, midi_in.port_name(p).unwrap());
            }
            print!("Please select input port: ");
            stdout().flush()?;
            let mut input = String::new();
            stdin().read_line(&mut input)?;
            in_ports
                .get(input.trim().parse::<usize>()?)
                .ok_or("invalid input port selected")?
        }
    };

    println!("\nOpening connection");
    //let in_port_name = midi_in.port_name(in_port)?;

    println!("notes array {:?}", voices);

    // Connection needs to be named to be kept alive.
    let _conn_in = midi_in.connect(
        in_port,
        "midir-read-input",
        move |stamp, message, _| {
            println!("{}: {:?} (len = {})", stamp, message, message.len());
            println!("notes {:?}", notes);
            match message.len() {
                2 => {
                    // Aftertouch?
                }
                3 => {
                    // Regular note data.
                    if message[2] == 0 {
                        // Note off.
                        //let note_data = notes.get(&message[1]);
                        let mut note_data = notes.remove(&message[1]).ok_or("No note found!?").unwrap();
                        let mut note_shared_vel = note_data.shared.lock().unwrap();
                        *note_shared_vel = 0.0;
                        println!("note data {:?}", note_data);
                        
                        //notes.remove(&message[1]);

                    }
                    else {
                        // Note on w/ velocity.
                        let shared = Arc::new(Mutex::new(1.0));
                        let note = wavetable_main(
                            440.0 * 2_f32.powf((message[1] as f32 - 69.0)/12.0),
                            message[2] as f32,
                            Arc::clone(&shared)
                        );
                        notes.insert(message[1], NoteData::new(note, shared));
                    }
                }
                _ => {
                    // Do nothing?
                }
            }
        },
        (),
    )?;

    //println!("Connection open, reading input from '{}'  (press enter to exit)", in_port_name);
    println!("Connection open (press enter to exit)");
    input.clear();
    stdin().read_line(&mut input)?; // Wait for enter/key press.
    
    println!("Closing connection");
    Ok(())
}