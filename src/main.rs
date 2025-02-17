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
use eframe::egui;
use std::sync::atomic::{AtomicU8, Ordering};

const ATTACK_MS: u128 = 2000;
const RELEASE_MS: u128 = 2000;

const WAVE_TYPE_SINE: u8 = 0;
const WAVE_TYPE_SAW: u8 = 1;
const WAVE_TYPE_SQUARE: u8 = 2;

const DEFAULT_ATTACK: u16 = 10;
const DEFAULT_DECAY: u16 = 600;
const DEFAULT_SUSTAIN: f32 = 0.2;
const DEFAULT_RELEASE: u16 = 800;

const SEMITONE_MIN: i32 = -36; // 3 octaves down
const SEMITONE_MAX: i32 = 36;  // 3 octaves up

const CENTS_MIN: f32 = -100.0;
const CENTS_MAX: f32 = 100.0;

const VOLUME_MIN: f32 = 0.0;
const VOLUME_MAX: f32 = 1.0;

const SAMPLE_RATE: u32 = 48000; // Higher sample rate for better quality
const ANTI_POP_SAMPLES: usize = 128; // Anti-pop buffer size

const FILTER_MODE_LOW: u8 = 0;
const FILTER_MODE_BAND: u8 = 1;
const FILTER_MODE_HIGH: u8 = 2;

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
    voice_amplitude: f32,
    attack: u16,
    decay: u16,
    sustain: f32,
    release: u16,
    shared: Arc<Mutex<f32>>,
    filter: StateVariableFilter,
    filter_cutoff: f32,
    filter_resonance: f32,
    filter_mode: u8,
}

impl WavetableOscillator {
    fn new(sample_rate: u32, wave_table: Vec<f32>, shared: Arc<Mutex<f32>>) -> WavetableOscillator {
        return WavetableOscillator {
            note_on: true,
            note_on_time: SystemTime::now(),
            note_off_time: SystemTime::now(),
            sample_rate: sample_rate,
            wave_table: wave_table,
            index: 0.0,
            index_increment: 0.0,
            amplitude: 1.0,
            voice_amplitude: 0.0,
            attack: 0,
            decay: 0,
            sustain: 0.0,
            release: 0,
            shared: shared,
            filter: StateVariableFilter::new(sample_rate),
            filter_cutoff: 1000.0,
            filter_resonance: 0.5,
            filter_mode: FILTER_MODE_LOW,
        }
    }

    fn set_attack(&mut self, attack: u16) {
        self.attack = attack;
    }

    fn set_decay(&mut self, decay: u16) {
        self.decay = decay;
    }

    fn set_sustain(&mut self, sustain: f32) {
        self.sustain = sustain;
    }

    fn set_release(&mut self, release: u16) {
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
            match self.note_on_time.elapsed() {
                Ok(elapsed) => {
                    // Attack amplitude envelope.
                    if self.attack > 0 {
                        if elapsed.as_millis() < self.attack as u128  {
                            amp *= elapsed.as_millis() as f32 / self.attack as f32;
                        }
                    }
                    if elapsed.as_millis() > self.attack as u128  {
                        // Attack done, start decay and sustain.
                        if self.decay > 0 {
                            let elapsed_since_attack = elapsed.as_millis() - self.attack as u128;
                            if elapsed_since_attack < self.decay as u128  {
                                amp -= (self.amplitude - self.amplitude * self.sustain) * (elapsed_since_attack as f32 / self.decay as f32);
                            }
                            else {
                                amp *= self.sustain;
                            }
                        }
                        else {
                            amp *= self.sustain;
                        }
                    }

                }
                Err(e) => {
                    println!("Error getting amplitude: {:?}", e);
                }
            }
            self.voice_amplitude = amp;
        }
        else {
            // Release amplitude envelope.
            amp = self.voice_amplitude;
            if self.release > 0 {
                match self.note_off_time.elapsed() {
                    Ok(elapsed) => {
                        //let release_length_ms: u128 = (RELEASE_MS as f32 * self.release) as u128;
                        if elapsed.as_millis() < self.release as u128 {
                            amp *= 1.0 - (elapsed.as_millis() as f32 / self.release as f32);
                        }
                        else {
                            amp = 0.0;
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

    fn set_filter_cutoff(&mut self, cutoff: f32) {
        self.filter_cutoff = cutoff.clamp(MIN_FREQ, MAX_FREQ);
        self.filter.set_cutoff(self.filter_cutoff);
    }

    fn set_filter_resonance(&mut self, resonance: f32) {
        self.filter_resonance = resonance.clamp(MIN_RESONANCE, MAX_RESONANCE);
        self.filter.set_resonance(self.filter_resonance);
    }

    fn set_filter_mode(&mut self, mode: u8) {
        self.filter_mode = mode;
        self.filter.set_mode(mode);
    }

    fn get_sample(&mut self) -> f32 {
        // Apply anti-pop envelope when starting or stopping
        if self.note_on {
            if *self.shared.lock().unwrap() == 0.0 {
                self.set_note_on(false);
            }
        }
        
        let sample = self.lerp();
        self.index += self.index_increment;
        self.index %= self.wave_table.len() as f32;
        
        // Get the main amplitude envelope
        let amp = self.get_amplitude();
        
        // Apply gentle anti-pop ramp at the very start and end
        let elapsed_samples = self.note_on_time.elapsed().unwrap().as_secs_f32() * self.sample_rate as f32;
        let start_ramp = if elapsed_samples < ANTI_POP_SAMPLES as f32 {
            elapsed_samples / ANTI_POP_SAMPLES as f32
        } else {
            1.0
        };
        
        let raw_sample = if !self.note_on {
            let release_samples = self.note_off_time.elapsed().unwrap().as_secs_f32() * self.sample_rate as f32;
            if release_samples < ANTI_POP_SAMPLES as f32 {
                amp * sample * (1.0 - release_samples / ANTI_POP_SAMPLES as f32) * start_ramp
            } else {
                amp * sample * start_ramp
            }
        } else {
            amp * sample * start_ramp
        };

        // Process through filter
        self.filter.process(raw_sample)
    }

    fn lerp(&mut self) -> f32 {
        let truncated_index = self.index as usize;
        let next_index = (truncated_index + 1) % self.wave_table.len();

        let next_index_weight = self.index - truncated_index as f32;
        let truncated_index_weight = 1.0 - next_index_weight;

        truncated_index_weight * self.wave_table[truncated_index] + 
            next_index_weight * self.wave_table[next_index]
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

const MIN_FREQ: f32 = 20.0;
const MAX_FREQ: f32 = 20000.0;
const MIN_RESONANCE: f32 = 0.0;
const MAX_RESONANCE: f32 = 1.0;

struct StateVariableFilter {
    cutoff: f32,
    resonance: f32,
    sample_rate: f32,
    low_pass: f32,
    band_pass: f32,
    high_pass: f32,
    last_input: f32,
    mode: u8,
}

impl StateVariableFilter {
    fn new(sample_rate: u32) -> Self {
        Self {
            cutoff: 1000.0,  // Default cutoff frequency
            resonance: 0.5,   // Default resonance
            sample_rate: sample_rate as f32,
            low_pass: 0.0,
            band_pass: 0.0,
            high_pass: 0.0,
            last_input: 0.0,
            mode: FILTER_MODE_LOW,
        }
    }

    fn set_cutoff(&mut self, cutoff: f32) {
        self.cutoff = cutoff.clamp(MIN_FREQ, MAX_FREQ);
    }

    fn set_resonance(&mut self, resonance: f32) {
        self.resonance = resonance.clamp(MIN_RESONANCE, MAX_RESONANCE);
    }

    fn set_mode(&mut self, mode: u8) {
        self.mode = mode;
    }

    fn process(&mut self, input: f32) -> f32 {
        // Calculate filter coefficients
        let f = 2.0 * std::f32::consts::PI * self.cutoff / self.sample_rate;
        let q = 1.0 - self.resonance;  // Higher resonance = lower damping
        
        // Clamp coefficients for stability
        let f = f.min(0.99);
        let q = q.max(0.01);

        // Process one sample
        self.high_pass = input - self.low_pass - q * self.band_pass;
        self.band_pass += f * self.high_pass;
        self.low_pass += f * self.band_pass;

        // Return output based on selected mode
        match self.mode {
            FILTER_MODE_LOW => self.low_pass,
            FILTER_MODE_BAND => self.band_pass,
            FILTER_MODE_HIGH => self.high_pass,
            _ => self.low_pass,
        }
    }
}

struct WtVoice {
    shared: Arc<Mutex<f32>>,
    wt_osc: WavetableOscillator,
    midi_port: midir::MidiInputPort,
    frequency: f32,
    amplitude: f32,
    note_on: bool,
    note_on_time: SystemTime,
    note_off_time: SystemTime,

}

impl WtVoice {
    fn new(_shared: Arc<Mutex<f32>>, _midi_port: midir::MidiInputPort, _frequency: f32, _amplitude: f32) {
        // Implementation pending
    }
}

#[derive(Clone)]
struct VoiceSettings {
    enabled: Arc<Mutex<bool>>,
    wave_type: Arc<AtomicU8>,
    semitones: Arc<Mutex<i32>>,
    cents: Arc<Mutex<f32>>,
    volume: Arc<Mutex<f32>>,
}

impl VoiceSettings {
    fn new(wave_type: u8, semitones: i32) -> Self {
        Self {
            enabled: Arc::new(Mutex::new(true)),
            wave_type: Arc::new(AtomicU8::new(wave_type)),
            semitones: Arc::new(Mutex::new(semitones)),
            cents: Arc::new(Mutex::new(0.0)),
            volume: Arc::new(Mutex::new(1.0)),
        }
    }

    fn is_enabled(&self) -> bool {
        *self.enabled.lock().unwrap()
    }

    fn set_enabled(&self, value: bool) {
        *self.enabled.lock().unwrap() = value;
    }

    fn get_cents(&self) -> f32 {
        *self.cents.lock().unwrap()
    }

    fn set_cents(&self, value: f32) {
        *self.cents.lock().unwrap() = value;
    }

    fn get_wave_type(&self) -> u8 {
        self.wave_type.load(Ordering::Relaxed)
    }

    fn set_wave_type(&self, value: u8) {
        self.wave_type.store(value, Ordering::Relaxed);
    }

    fn get_semitones(&self) -> i32 {
        *self.semitones.lock().unwrap()
    }

    fn set_semitones(&self, value: i32) {
        *self.semitones.lock().unwrap() = value;
    }

    fn get_volume(&self) -> f32 {
        *self.volume.lock().unwrap()
    }

    fn set_volume(&self, value: f32) {
        *self.volume.lock().unwrap() = value;
    }
}

#[derive(Clone)]
struct SynthSettings {
    attack: Arc<Mutex<u16>>,
    decay: Arc<Mutex<u16>>,
    sustain: Arc<Mutex<f32>>,
    release: Arc<Mutex<u16>>,
    filter_cutoff: Arc<Mutex<f32>>,
    filter_resonance: Arc<Mutex<f32>>,
    filter_mode: Arc<AtomicU8>,
    voice1: VoiceSettings,
    voice2: VoiceSettings,
    voice3: VoiceSettings,
}

impl Default for SynthSettings {
    fn default() -> Self {
        Self {
            attack: Arc::new(Mutex::new(DEFAULT_ATTACK)),
            decay: Arc::new(Mutex::new(DEFAULT_DECAY)),
            sustain: Arc::new(Mutex::new(DEFAULT_SUSTAIN)),
            release: Arc::new(Mutex::new(DEFAULT_RELEASE)),
            filter_cutoff: Arc::new(Mutex::new(1000.0)),
            filter_resonance: Arc::new(Mutex::new(0.5)),
            filter_mode: Arc::new(AtomicU8::new(FILTER_MODE_LOW)),
            voice1: VoiceSettings::new(WAVE_TYPE_SAW, 0),     // Default to saw, no transpose
            voice2: VoiceSettings::new(WAVE_TYPE_SQUARE, -12), // Default to square, -1 octave
            voice3: VoiceSettings::new(WAVE_TYPE_SINE, 12),   // Default to sine, +1 octave
        }
    }
}

impl SynthSettings {
    fn get_attack(&self) -> u16 {
        *self.attack.lock().unwrap()
    }

    fn get_decay(&self) -> u16 {
        *self.decay.lock().unwrap()
    }

    fn get_sustain(&self) -> f32 {
        *self.sustain.lock().unwrap()
    }

    fn get_release(&self) -> u16 {
        *self.release.lock().unwrap()
    }

    fn set_attack(&self, value: u16) {
        *self.attack.lock().unwrap() = value;
    }

    fn set_decay(&self, value: u16) {
        *self.decay.lock().unwrap() = value;
    }

    fn set_sustain(&self, value: f32) {
        *self.sustain.lock().unwrap() = value;
    }

    fn set_release(&self, value: u16) {
        *self.release.lock().unwrap() = value;
    }

    fn get_filter_cutoff(&self) -> f32 {
        *self.filter_cutoff.lock().unwrap()
    }

    fn get_filter_resonance(&self) -> f32 {
        *self.filter_resonance.lock().unwrap()
    }

    fn set_filter_cutoff(&self, value: f32) {
        *self.filter_cutoff.lock().unwrap() = value.clamp(MIN_FREQ, MAX_FREQ);
    }

    fn set_filter_resonance(&self, value: f32) {
        *self.filter_resonance.lock().unwrap() = value.clamp(MIN_RESONANCE, MAX_RESONANCE);
    }

    fn get_filter_mode(&self) -> u8 {
        self.filter_mode.load(Ordering::Relaxed)
    }

    fn set_filter_mode(&self, mode: u8) {
        self.filter_mode.store(mode, Ordering::Relaxed);
    }
}

struct SynthUI {
    settings: SynthSettings,
    midi_thread: Option<std::thread::JoinHandle<()>>,
    settings_sender: mpsc::Sender<SynthSettings>,
}

impl SynthUI {
    fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        let (settings_sender, settings_receiver) = mpsc::channel();
        
        // Start MIDI handling in a separate thread
        let midi_thread = Some(std::thread::spawn(move || {
            if let Err(err) = run(settings_receiver) {
                println!("Error in MIDI thread: {}", err);
            }
        }));

        Self {
            settings: SynthSettings::default(),
            midi_thread,
            settings_sender,
        }
    }
}

impl eframe::App for SynthUI {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Synthesizer Controls");
            
            // Helper function to create voice controls
            let add_voice_controls = |ui: &mut egui::Ui, voice: &VoiceSettings, voice_name: &str| {
                ui.group(|ui| {
                    ui.horizontal(|ui| {
                        ui.heading(voice_name);
                        let mut enabled = voice.is_enabled();
                        if ui.checkbox(&mut enabled, "Enabled").changed() {
                            voice.set_enabled(enabled);
                            let _ = self.settings_sender.send(self.settings.clone());
                        }
                    });

                    if voice.is_enabled() {
                        ui.horizontal(|ui| {
                            ui.label("Volume:");
                            let mut volume = voice.get_volume();
                            if ui.add(egui::Slider::new(&mut volume, VOLUME_MIN..=VOLUME_MAX)
                                .text("vol")).changed() {
                                voice.set_volume(volume);
                                let _ = self.settings_sender.send(self.settings.clone());
                            }
                        });

                        ui.horizontal(|ui| {
                            ui.label("Wave Type:");
                            let mut current_wave = voice.get_wave_type();
                            if ui.radio_value(&mut current_wave, WAVE_TYPE_SINE, "Sine").clicked() ||
                               ui.radio_value(&mut current_wave, WAVE_TYPE_SAW, "Saw").clicked() ||
                               ui.radio_value(&mut current_wave, WAVE_TYPE_SQUARE, "Square").clicked() {
                                voice.set_wave_type(current_wave);
                                let _ = self.settings_sender.send(self.settings.clone());
                            }
                        });
                        
                        ui.horizontal(|ui| {
                            ui.label("Semitones:");
                            let mut semitones = voice.get_semitones();
                            if ui.add(egui::Slider::new(&mut semitones, SEMITONE_MIN..=SEMITONE_MAX)).changed() {
                                voice.set_semitones(semitones);
                                let _ = self.settings_sender.send(self.settings.clone());
                            }
                        });

                        ui.horizontal(|ui| {
                            ui.label("Cents:");
                            let mut cents = voice.get_cents();
                            if ui.add(egui::Slider::new(&mut cents, CENTS_MIN..=CENTS_MAX)).changed() {
                                voice.set_cents(cents);
                                let _ = self.settings_sender.send(self.settings.clone());
                            }
                        });
                    }
                });
            };

            add_voice_controls(ui, &self.settings.voice1, "Voice 1");
            ui.add_space(10.0);
            add_voice_controls(ui, &self.settings.voice2, "Voice 2");
            ui.add_space(10.0);
            add_voice_controls(ui, &self.settings.voice3, "Voice 3");
            ui.add_space(10.0);

            // Filter controls
            ui.group(|ui| {
                ui.heading("Filter");
                
                // Add filter mode selection
                ui.horizontal(|ui| {
                    ui.label("Mode:");
                    let mut current_mode = self.settings.get_filter_mode();
                    if ui.radio_value(&mut current_mode, FILTER_MODE_LOW, "Low Pass").clicked() ||
                       ui.radio_value(&mut current_mode, FILTER_MODE_BAND, "Band Pass").clicked() ||
                       ui.radio_value(&mut current_mode, FILTER_MODE_HIGH, "High Pass").clicked() {
                        self.settings.set_filter_mode(current_mode);
                        let _ = self.settings_sender.send(self.settings.clone());
                    }
                });

                ui.horizontal(|ui| {
                    ui.label("Cutoff:");
                    let mut cutoff = self.settings.get_filter_cutoff();
                    if ui.add(
                        egui::Slider::new(&mut cutoff, MIN_FREQ..=MAX_FREQ)
                            .logarithmic(true)
                            .text("Hz")
                    ).changed() {
                        self.settings.set_filter_cutoff(cutoff);
                        let _ = self.settings_sender.send(self.settings.clone());
                    }
                });
                
                ui.horizontal(|ui| {
                    ui.label("Resonance:");
                    let mut resonance = self.settings.get_filter_resonance();
                    if ui.add(egui::Slider::new(&mut resonance, MIN_RESONANCE..=MAX_RESONANCE)).changed() {
                        self.settings.set_filter_resonance(resonance);
                        let _ = self.settings_sender.send(self.settings.clone());
                    }
                });
            });

            ui.add_space(10.0);

            // ADSR controls
            ui.group(|ui| {
                ui.heading("Envelope");
                ui.horizontal(|ui| {
                    ui.label("Attack:");
                    let mut attack = self.settings.get_attack();
                    if ui.add(egui::Slider::new(&mut attack, 0..=2000).text("ms")).changed() {
                        self.settings.set_attack(attack);
                        let _ = self.settings_sender.send(self.settings.clone());
                    }
                });
                
                ui.horizontal(|ui| {
                    ui.label("Decay:");
                    let mut decay = self.settings.get_decay();
                    if ui.add(egui::Slider::new(&mut decay, 0..=2000).text("ms")).changed() {
                        self.settings.set_decay(decay);
                        let _ = self.settings_sender.send(self.settings.clone());
                    }
                });
                
                ui.horizontal(|ui| {
                    ui.label("Sustain:");
                    let mut sustain = self.settings.get_sustain();
                    if ui.add(egui::Slider::new(&mut sustain, 0.0..=1.0)).changed() {
                        self.settings.set_sustain(sustain);
                        let _ = self.settings_sender.send(self.settings.clone());
                    }
                });
                
                ui.horizontal(|ui| {
                    ui.label("Release:");
                    let mut release = self.settings.get_release();
                    if ui.add(egui::Slider::new(&mut release, 0..=2000).text("ms")).changed() {
                        self.settings.set_release(release);
                        let _ = self.settings_sender.send(self.settings.clone());
                    }
                });
            });
        });
    }
}

fn wavetable_sine() -> Vec<f32> {
    let wave_table_size = 2048;
    let mut wave_table: Vec<f32> = Vec::with_capacity(wave_table_size);
    for n in 0..wave_table_size {
        wave_table.push((2.0 * std::f32::consts::PI * n as f32 / wave_table_size as f32).sin());
    }
    wave_table
}

fn wavetable_saw() -> Vec<f32> {
    let wave_table_size = 2048;
    let mut wave_table: Vec<f32> = Vec::with_capacity(wave_table_size);
    for n in 0..wave_table_size {
        wave_table.push(1.0 - 2.0 * (n as f32 / wave_table_size as f32));
    }
    wave_table
}

fn wavetable_square() -> Vec<f32> {
    let wave_table_size = 2048;
    let mut wave_table: Vec<f32> = Vec::with_capacity(wave_table_size);
    for n in 0..wave_table_size {
        let mut val: f32 = 1.0;
        if n < wave_table_size / 2 {
            val = -1.0;
        }
        wave_table.push(val);
    }
    wave_table
}

fn wavetable_main(
    settings: SynthSettings,
    frequency: f32,
    velocity: f32,
    shared: Arc<Mutex<f32>>,
    wave_type: u8,
) -> thread::JoinHandle<()> {
    let note = std::thread::spawn(move || {
        let wave_table: Vec<f32> = match wave_type {
            WAVE_TYPE_SINE => wavetable_sine(),
            WAVE_TYPE_SAW => wavetable_saw(),
            WAVE_TYPE_SQUARE => wavetable_square(),
            _ => wavetable_sine(),
        };
        
        let mut oscillator = WavetableOscillator::new(SAMPLE_RATE, wave_table, Arc::clone(&shared));
        oscillator.set_frequency(frequency);
        oscillator.set_amplitude(velocity/127.0);
        
        oscillator.set_attack(settings.get_attack());
        oscillator.set_decay(settings.get_decay());
        oscillator.set_sustain(settings.get_sustain());
        oscillator.set_release(settings.get_release());
        
        // Apply filter settings
        oscillator.set_filter_cutoff(settings.get_filter_cutoff());
        oscillator.set_filter_resonance(settings.get_filter_resonance());
        oscillator.set_filter_mode(settings.get_filter_mode());

        let (_stream, stream_handle) = OutputStream::try_default().unwrap();
        let _result = stream_handle.play_raw(oscillator.convert_samples());
        
        // Use a shorter sleep interval for better responsiveness
        while *shared.lock().unwrap() > 0.0 {
            thread::sleep(Duration::from_micros(100));
        }
        thread::sleep(Duration::from_millis(settings.get_release() as u64));
    });
    note
}

fn main() {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([320.0, 240.0]),
        ..Default::default()
    };
    
    eframe::run_native(
        "Synthesizer Controls",
        options,
        Box::new(|cc| Box::new(SynthUI::new(cc))),
    )
    .unwrap();
}

fn run(settings_receiver: mpsc::Receiver<SynthSettings>) -> Result<(), Box<dyn Error>> {
    let mut input = String::new();
    let mut voice1: HashMap<u8, NoteData> = HashMap::with_capacity(16);
    let mut voice2: HashMap<u8, NoteData> = HashMap::with_capacity(16);
    let mut voice3: HashMap<u8, NoteData> = HashMap::with_capacity(16);

    let mut midi_in = MidiInput::new("midi_read_fx")?;
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
    let mut current_settings = SynthSettings::default();

    let _conn_in = midi_in.connect(
        in_port,
        "midir-read-input",
        move |_stamp, message, _| {
            // Update settings if new ones are available
            while let Ok(new_settings) = settings_receiver.try_recv() {
                current_settings = new_settings;
            }

            match message.len() {
                3 => {
                    if message[0] == 0x80 || (message[0] == 0x90 && message[2] == 0) {
                        // Note off - process immediately for responsive release
                        let mut voice_maps = [&mut voice1, &mut voice2, &mut voice3];
                        for voice_map in voice_maps.iter_mut() {
                            if let Some(note_data) = voice_map.remove(&message[1]) {
                                let mut note_shared_vel = note_data.shared.lock().unwrap();
                                *note_shared_vel = 0.0;
                            }
                        }
                    }
                    else if message[0] == 0x90 {
                        // Helper function to calculate frequency with cents
                        let calc_freq = |semitones: i32, cents: f32| -> f32 {
                            let total_semitones = semitones as f32 + (cents / 100.0);
                            440.0 * 2_f32.powf((message[1] as f32 - 69.0 + total_semitones)/12.0)
                        };

                        // Create voices if enabled
                        if current_settings.voice1.is_enabled() {
                            let shared = Arc::new(Mutex::new(1.0));
                            let note = wavetable_main(
                                current_settings.clone(),
                                calc_freq(
                                    current_settings.voice1.get_semitones(),
                                    current_settings.voice1.get_cents()
                                ),
                                message[2] as f32 * current_settings.voice1.get_volume(),
                                Arc::clone(&shared),
                                current_settings.voice1.get_wave_type()
                            );
                            voice1.insert(message[1], NoteData::new(note, shared));
                        }

                        if current_settings.voice2.is_enabled() {
                            let shared = Arc::new(Mutex::new(1.0));
                            let note = wavetable_main(
                                current_settings.clone(),
                                calc_freq(
                                    current_settings.voice2.get_semitones(),
                                    current_settings.voice2.get_cents()
                                ),
                                message[2] as f32 * current_settings.voice2.get_volume(),
                                Arc::clone(&shared),
                                current_settings.voice2.get_wave_type()
                            );
                            voice2.insert(message[1], NoteData::new(note, shared));
                        }

                        if current_settings.voice3.is_enabled() {
                            let shared = Arc::new(Mutex::new(1.0));
                            let note = wavetable_main(
                                current_settings.clone(),
                                calc_freq(
                                    current_settings.voice3.get_semitones(),
                                    current_settings.voice3.get_cents()
                                ),
                                message[2] as f32 * current_settings.voice3.get_volume(),
                                Arc::clone(&shared),
                                current_settings.voice3.get_wave_type()
                            );
                            voice3.insert(message[1], NoteData::new(note, shared));
                        }
                    }
                }
                _ => {}
            }
        },
        (),
    )?;

    println!("Connection open (press enter to exit)");
    input.clear();
    stdin().read_line(&mut input)?;
    println!("Closing connection");
    Ok(())
}