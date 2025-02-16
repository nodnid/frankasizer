// main.rs

use rodio::{OutputStream, source::Source};
use std::collections::HashMap;
use std::error::Error;
use std::io::{stdin, stdout, Write};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, SystemTime};
use midir::Ignore;
use midir::MidiInput;

const ATTACK_MS: u128 = 2000;
const RELEASE_MS: u128 = 2000;

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
    note_on: bool,
    note_on_time: SystemTime,
    note_off_time: SystemTime,
    sample_rate: u32,
    wave_table: Vec<f32>,
    index: f32,
    index_increment: f32,
    amplitude: f32,
    attack: f32,
    decay: f32,
    sustain: f32,
    release: f32,
}

impl WavetableOscillator {
    fn new(sample_rate: u32, wave_table: Vec<f32>) -> WavetableOscillator {
        return WavetableOscillator {
            note_on: true,
            note_on_time: SystemTime::now(),
            note_off_time: SystemTime::now(),
            sample_rate: sample_rate,
            wave_table: wave_table,
            index: 0.0,
            index_increment: 0.0,
            amplitude: 1.0,
            attack: 0.0,
            decay: 0.0,
            sustain: 0.0,
            release: 0.0,
        }
    }

    fn set_attack(&mut self, attack: f32) {
        self.attack = attack;
    }

    fn set_decay(&mut self, decay: f32) {
        self.decay = decay;
    }

    fn set_sustain(&mut self, sustain: f32) {
        self.sustain = sustain;
    }

    fn set_release(&mut self, release: f32) {
        self.release = release;
    }

    fn set_frequency(&mut self, frequency: f32) {
        self.index_increment = frequency * self.wave_table.len() as f32 / self.sample_rate as f32;
    }

    fn set_amplitude(&mut self, amplitude: f32) {
        self.amplitude = amplitude;
    }

    fn set_note_on(&mut self, note_on: bool) {
        self.note_on = note_on;
        if !note_on {
            self.note_off_time = SystemTime::now();
        }
    }

    fn get_amplitude(&mut self) -> f32 {
        let mut amp: f32 = self.amplitude;
        if self.note_on {
            // Attack amplitude envelope.
            if self.attack > 0.0 {
                match self.note_on_time.elapsed() {
                    Ok(elapsed) => {
                        let attack_length_ms: u128 = (ATTACK_MS as f32 * self.attack) as u128;
                        if elapsed.as_millis() < attack_length_ms  {
                            amp *= (elapsed.as_millis() as f32 / attack_length_ms as f32);
                        }
                    }
                    Err(e) => {
                        println!("Error getting amplitude: {:?}", e);
                    }
                }
            }
        }
        else {
            // Release amplitude envelope.
            if self.release > 0.0 {
                match self.note_off_time.elapsed() {
                    Ok(elapsed) => {
                        let release_length_ms: u128 = (RELEASE_MS as f32 * self.release) as u128;
                        if elapsed.as_millis() < release_length_ms  {
                            amp *= 1.0 - (elapsed.as_millis() as f32 / release_length_ms as f32);
                        }
                    }
                    Err(e) => {
                        println!("Error getting amplitude: {:?}", e);
                    }
                }
            }

        }

        return amp;
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

    fn lerp_sine(&mut self) -> f32 {
        let truncated_index = self.index as usize;
        let next_index = (truncated_index + 1) % self.wave_table.len();

        let next_index_weight = self.index - truncated_index as f32;
        let truncated_index_weight = 1.0 - next_index_weight;

        return self.get_amplitude() * (truncated_index_weight * self.wave_table[truncated_index] + next_index_weight * self.wave_table[next_index]);
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
        // Set attack, delay, sustain, release.
        oscillator.set_attack(0.5);
        oscillator.set_release(1.0);

        let (_stream, stream_handle) = OutputStream::try_default().unwrap();
        let _result = stream_handle.play_raw(oscillator.convert_samples());
        while *shared.lock().unwrap() > 0.0 {}
        //oscillator.set_note_on(false);
        //std::thread::sleep(std::time::Duration::from_millis(RELEASE_MS as u64));
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
    // Connection needs to be named to be kept alive.
    let _conn_in = midi_in.connect(
        in_port,
        "midir-read-input",
        move |stamp, message, _| {
            println!("{}: {:?} (len = {})", stamp, message, message.len());
            match message.len() {
                2 => {
                    // Aftertouch?
                }
                3 => {
                    // Regular note data.
                    if message[0] == 0x80 || (message[0] == 0x90 && message[2] == 0) {
                        // Note off.
                        let mut note_data = notes.remove(&message[1]).ok_or("No note found!?").unwrap();
                        let mut note_shared_vel = note_data.shared.lock().unwrap();
                        *note_shared_vel = 0.0;
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