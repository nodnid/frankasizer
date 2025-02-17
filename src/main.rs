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
use rand::Rng;

mod constants {
    // Envelope constants
    pub const ATTACK_MS: u128 = 2000;
    pub const RELEASE_MS: u128 = 2000;
    pub const DEFAULT_ATTACK: u16 = 10;
    pub const DEFAULT_DECAY: u16 = 600;
    pub const DEFAULT_SUSTAIN: f32 = 0.2;
    pub const DEFAULT_RELEASE: u16 = 800;

    // Wave type constants
    pub const WAVE_TYPE_SINE: u8 = 0;
    pub const WAVE_TYPE_SAW: u8 = 1;
    pub const WAVE_TYPE_SQUARE: u8 = 2;
    pub const WAVE_TYPE_TRIANGLE: u8 = 3;
    pub const WAVE_TYPE_NOISE: u8 = 4;

    // Pitch constants
    pub const SEMITONE_MIN: i32 = -36; // 3 octaves down
    pub const SEMITONE_MAX: i32 = 36;  // 3 octaves up
    pub const CENTS_MIN: f32 = -100.0;
    pub const CENTS_MAX: f32 = 100.0;

    // Volume constants
    pub const VOLUME_MIN: f32 = 0.0;
    pub const VOLUME_MAX: f32 = 1.0;

    // Audio constants
    pub const SAMPLE_RATE: u32 = 48000;
    pub const ANTI_POP_SAMPLES: usize = 128;
    pub const WAVETABLE_SIZE: usize = 2048;

    // Filter constants
    pub const FILTER_MODE_LOW: u8 = 0;
    pub const FILTER_MODE_BAND: u8 = 1;
    pub const FILTER_MODE_HIGH: u8 = 2;
    pub const MIN_FREQ: f32 = 20.0;
    pub const MAX_FREQ: f32 = 20000.0;
    pub const MIN_RESONANCE: f32 = 0.0;
    pub const MAX_RESONANCE: f32 = 1.0;

    // LFO constants
    pub const LFO_MIN_FREQ: f32 = 0.1;
    pub const LFO_MAX_FREQ: f32 = 20.0;

    // Mode constants
    pub const MODE_POLY: u8 = 0;
    pub const MODE_MONO: u8 = 1;

    // Effect constants
    pub const EFFECT_NONE: u8 = 0;
    pub const EFFECT_DISTORTION: u8 = 1;

    // Distortion constants
    pub const DRIVE_MIN: f32 = 1.0;
    pub const DRIVE_MAX: f32 = 50.0;
}

use constants::*;

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
    lfo_wave_table: Vec<f32>,
    lfo_index: f32,
    lfo_freq: f32,
    lfo_amount: f32,
    target_frequency: f32,
    current_frequency: f32,
    portamento_time: f32,
    distortion1: Option<Distortion>,
    distortion2: Option<Distortion>,
    effect1_type: u8,
    effect2_type: u8,
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
            lfo_wave_table: Vec::new(),
            lfo_index: 0.0,
            lfo_freq: 0.0,
            lfo_amount: 0.0,
            target_frequency: 0.0,
            current_frequency: 0.0,
            portamento_time: 0.0,
            distortion1: None,
            distortion2: None,
            effect1_type: EFFECT_NONE,
            effect2_type: EFFECT_NONE,
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
        self.target_frequency = frequency;
        if self.portamento_time == 0.0 {
            self.current_frequency = frequency;
            self.update_index_increment();
        }
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
        // Process portamento before generating the sample
        self.process_portamento();

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
        let filtered_sample = self.filter.process(raw_sample);

        // Apply LFO modulation to filter cutoff
        let lfo_value = self.get_lfo_sample();
        let modulated_cutoff = self.filter_cutoff * (1.0 + lfo_value * self.lfo_amount);
        self.filter.set_cutoff(modulated_cutoff);

        // Apply effects chain
        let mut processed = filtered_sample;

        // Effect 1
        processed = match self.effect1_type {
            EFFECT_DISTORTION => {
                if let Some(dist) = &self.distortion1 {
                    dist.process(processed)
                } else {
                    processed
                }
            }
            _ => processed,
        };

        // Effect 2
        processed = match self.effect2_type {
            EFFECT_DISTORTION => {
                if let Some(dist) = &self.distortion2 {
                    dist.process(processed)
                } else {
                    processed
                }
            }
            _ => processed,
        };

        // Ensure we're not silencing the signal
        processed
    }

    fn lerp(&mut self) -> f32 {
        let truncated_index = self.index as usize;
        let next_index = (truncated_index + 1) % self.wave_table.len();

        let next_index_weight = self.index - truncated_index as f32;
        let truncated_index_weight = 1.0 - next_index_weight;

        truncated_index_weight * self.wave_table[truncated_index] + 
            next_index_weight * self.wave_table[next_index]
    }

    fn get_lfo_sample(&mut self) -> f32 {
        if self.lfo_wave_table.is_empty() || self.lfo_amount == 0.0 {
            return 0.0;
        }
        
        let index = self.lfo_index as usize & (self.lfo_wave_table.len() - 1);
        let sample = self.lfo_wave_table[index];
        
        self.lfo_index = (self.lfo_index + self.lfo_freq) % self.lfo_wave_table.len() as f32;
        sample
    }

    fn process_portamento(&mut self) {
        if self.current_frequency != self.target_frequency && self.portamento_time > 0.0 {
            // Pre-calculate the rate once
            let rate = (self.target_frequency - self.current_frequency) 
                      / (self.portamento_time * self.sample_rate as f32);
            
            let new_freq = self.current_frequency + rate;
            
            // Simpler bounds check
            if rate > 0.0 {
                self.current_frequency = new_freq.min(self.target_frequency);
            } else {
                self.current_frequency = new_freq.max(self.target_frequency);
            }
            
            // Only update index_increment when frequency actually changes
            self.index_increment = self.current_frequency * self.wave_table.len() as f32 
                                 / self.sample_rate as f32;
        }
    }
    
    fn update_index_increment(&mut self) {
        self.index_increment = self.current_frequency * self.wave_table.len() as f32 
                              / self.sample_rate as f32;
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
struct EffectSettings {
    effect_type: Arc<AtomicU8>,
    drive: Arc<Mutex<f32>>,
    mix: Arc<Mutex<f32>>,
}

impl Default for EffectSettings {
    fn default() -> Self {
        Self {
            effect_type: Arc::new(AtomicU8::new(EFFECT_NONE)),
            drive: Arc::new(Mutex::new(1.0)),
            mix: Arc::new(Mutex::new(0.5)),
        }
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
    max_polyphony: Arc<Mutex<u8>>,
    portamento_time: Arc<Mutex<f32>>,
    lfo_freq: Arc<Mutex<f32>>,
    lfo_amount: Arc<Mutex<f32>>,
    play_mode: Arc<AtomicU8>,
    effect1: EffectSettings,
    effect2: EffectSettings,
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
            max_polyphony: Arc::new(Mutex::new(8)),
            portamento_time: Arc::new(Mutex::new(0.0)),
            lfo_freq: Arc::new(Mutex::new(0.0)),
            lfo_amount: Arc::new(Mutex::new(0.0)),
            play_mode: Arc::new(AtomicU8::new(MODE_POLY)),
            effect1: EffectSettings::default(),
            effect2: EffectSettings::default(),
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

    fn get_portamento_time(&self) -> f32 {
        *self.portamento_time.lock().unwrap()
    }

    fn set_portamento_time(&self, value: f32) {
        *self.portamento_time.lock().unwrap() = value;
    }

    fn get_lfo_freq(&self) -> f32 {
        *self.lfo_freq.lock().unwrap()
    }

    fn set_lfo_freq(&self, value: f32) {
        *self.lfo_freq.lock().unwrap() = value.clamp(LFO_MIN_FREQ, LFO_MAX_FREQ);
    }

    fn get_lfo_amount(&self) -> f32 {
        *self.lfo_amount.lock().unwrap()
    }

    fn set_lfo_amount(&self, value: f32) {
        *self.lfo_amount.lock().unwrap() = value.clamp(0.0, 1.0);
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

    fn render_voice_controls(
        ui: &mut egui::Ui,
        voice: &VoiceSettings,
        voice_name: &str,
        settings: &SynthSettings,
        settings_sender: &mpsc::Sender<SynthSettings>,
    ) {
        ui.group(|ui| {
            // Voice header with enable toggle
            ui.horizontal(|ui| {
                let mut enabled = voice.is_enabled();
                if ui.checkbox(&mut enabled, "").changed() {
                    voice.set_enabled(enabled);
                    let _ = settings_sender.send(settings.clone());
                }
                ui.heading(voice_name);
            });

            if voice.is_enabled() {
                ui.add_space(5.0);

                // Wave type selection with custom styling
                ui.horizontal(|ui| {
                    ui.label("Wave:");
                    let mut current_wave = voice.get_wave_type();
                    ui.selectable_value(&mut current_wave, WAVE_TYPE_SINE, "Sine");
                    ui.selectable_value(&mut current_wave, WAVE_TYPE_SAW, "Saw");
                    ui.selectable_value(&mut current_wave, WAVE_TYPE_SQUARE, "Square");
                    ui.selectable_value(&mut current_wave, WAVE_TYPE_TRIANGLE, "Triangle");
                    ui.selectable_value(&mut current_wave, WAVE_TYPE_NOISE, "Noise");
                    if current_wave != voice.get_wave_type() {
                        voice.set_wave_type(current_wave);
                        let _ = settings_sender.send(settings.clone());
                    }
                });

                ui.add_space(5.0);

                // Volume control
                ui.horizontal(|ui| {
                    ui.label("Volume:");
                    let mut volume = voice.get_volume();
                    if ui.add(
                        egui::Slider::new(&mut volume, VOLUME_MIN..=VOLUME_MAX)
                            .text("")
                    ).changed() {
                        voice.set_volume(volume);
                        let _ = settings_sender.send(settings.clone());
                    }
                });

                ui.add_space(5.0);

                // Pitch controls
                egui::Grid::new(format!("{}_pitch_grid", voice_name))
                    .spacing([10.0, 5.0])
                    .show(ui, |ui| {
                        // Semitones
                        ui.label("Semitones:");
                        let mut semitones = voice.get_semitones();
                        if ui.add(
                            egui::Slider::new(&mut semitones, SEMITONE_MIN..=SEMITONE_MAX)
                                .text("")
                        ).changed() {
                            voice.set_semitones(semitones);
                            let _ = settings_sender.send(settings.clone());
                        }
                        ui.end_row();

                        // Cents
                        ui.label("Cents:");
                        let mut cents = voice.get_cents();
                        if ui.add(
                            egui::Slider::new(&mut cents, CENTS_MIN..=CENTS_MAX)
                                .text("")
                        ).changed() {
                            voice.set_cents(cents);
                            let _ = settings_sender.send(settings.clone());
                        }
                        ui.end_row();
                    });
            }
        });
    }

    fn render_effect_controls(
        ui: &mut egui::Ui,
        effect: &EffectSettings,
        effect_name: &str,
        settings: &SynthSettings,
        settings_sender: &mpsc::Sender<SynthSettings>,
    ) {
        ui.group(|ui| {
            ui.heading(effect_name);
            ui.add_space(5.0);

            // Effect type selector
            ui.horizontal(|ui| {
                ui.label("Type:");
                let mut current_effect = effect.effect_type.load(Ordering::Relaxed);
                ui.selectable_value(&mut current_effect, EFFECT_NONE, "None");
                ui.selectable_value(&mut current_effect, EFFECT_DISTORTION, "Distortion");
                if current_effect != effect.effect_type.load(Ordering::Relaxed) {
                    effect.effect_type.store(current_effect, Ordering::Relaxed);
                    let _ = settings_sender.send(settings.clone());
                }
            });

            match effect.effect_type.load(Ordering::Relaxed) {
                EFFECT_DISTORTION => {
                    ui.add_space(5.0);
                    
                    // Distortion controls
                    ui.horizontal(|ui| {
                        ui.label("Drive:");
                        let mut drive = *effect.drive.lock().unwrap();
                        if ui.add(
                            egui::Slider::new(&mut drive, DRIVE_MIN..=DRIVE_MAX)
                                .logarithmic(true)
                        ).changed() {
                            *effect.drive.lock().unwrap() = drive;
                            let _ = settings_sender.send(settings.clone());
                        }
                    });

                    ui.horizontal(|ui| {
                        ui.label("Mix:");
                        let mut mix = *effect.mix.lock().unwrap();
                        if ui.add(
                            egui::Slider::new(&mut mix, 0.0..=1.0)
                        ).changed() {
                            *effect.mix.lock().unwrap() = mix;
                            let _ = settings_sender.send(settings.clone());
                        }
                    });
                }
                _ => {}
            }
        });
    }

    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Set up a dark theme
        ctx.set_visuals(egui::Visuals::dark());

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.vertical_centered(|ui| {
                ui.heading("WTSynth");
                ui.add_space(10.0);
            });

            // Main container with spacing
            egui::Grid::new("synth_grid")
                .spacing([20.0, 20.0])
                .show(ui, |ui| {
                    // Left column - Voices
                    ui.vertical(|ui| {
                        ui.group(|ui| {
                            ui.set_width(300.0);
                            ui.heading("Voices");
                            ui.add_space(10.0);
                            
                            let settings = &self.settings;
                            let settings_sender = &self.settings_sender;
                            
                            Self::render_voice_controls(ui, &settings.voice1, "Voice 1", settings, settings_sender);
                            ui.add_space(5.0);
                            Self::render_voice_controls(ui, &settings.voice2, "Voice 2", settings, settings_sender);
                            ui.add_space(5.0);
                            Self::render_voice_controls(ui, &settings.voice3, "Voice 3", settings, settings_sender);
                        });
                    });

                    // Right column - Filter and Envelope
                    ui.vertical(|ui| {
                        // Filter section
                        ui.group(|ui| {
                            ui.set_width(300.0);
                            ui.heading("Filter");
                            ui.add_space(10.0);
                            
                            // Filter mode selection with custom styling
                            ui.horizontal(|ui| {
                                ui.label("Mode:");
                                let mut current_mode = self.settings.get_filter_mode();
                                ui.selectable_value(&mut current_mode, FILTER_MODE_LOW, "Low Pass");
                                ui.selectable_value(&mut current_mode, FILTER_MODE_BAND, "Band Pass");
                                ui.selectable_value(&mut current_mode, FILTER_MODE_HIGH, "High Pass");
                                if current_mode != self.settings.get_filter_mode() {
                                    self.settings.set_filter_mode(current_mode);
                                    let _ = self.settings_sender.send(self.settings.clone());
                                }
                            });

                            ui.add_space(5.0);

                            // Cutoff with logarithmic slider
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

                            ui.add_space(5.0);
                            
                            // Resonance with visual feedback
                            ui.horizontal(|ui| {
                                ui.label("Resonance:");
                                let mut resonance = self.settings.get_filter_resonance();
                                if ui.add(
                                    egui::Slider::new(&mut resonance, MIN_RESONANCE..=MAX_RESONANCE)
                                        .text("")
                                ).changed() {
                                    self.settings.set_filter_resonance(resonance);
                                    let _ = self.settings_sender.send(self.settings.clone());
                                }
                            });
                        });

                        ui.add_space(10.0);

                        // ADSR Envelope section
                        ui.group(|ui| {
                            ui.set_width(300.0);
                            ui.heading("Envelope");
                            ui.add_space(10.0);

                            let envelope_response = egui::Grid::new("envelope_grid")
                                .spacing([10.0, 10.0])
                                .show(ui, |ui| {
                                    // Attack
                                    ui.label("Attack:");
                                    let mut attack = self.settings.get_attack();
                                    let attack_changed = ui.add(
                                        egui::Slider::new(&mut attack, 0..=2000)
                                            .text("ms")
                                    ).changed();
                                    ui.end_row();

                                    // Decay
                                    ui.label("Decay:");
                                    let mut decay = self.settings.get_decay();
                                    let decay_changed = ui.add(
                                        egui::Slider::new(&mut decay, 0..=2000)
                                            .text("ms")
                                    ).changed();
                                    ui.end_row();

                                    // Sustain
                                    ui.label("Sustain:");
                                    let mut sustain = self.settings.get_sustain();
                                    let sustain_changed = ui.add(
                                        egui::Slider::new(&mut sustain, 0.0..=1.0)
                                            .text("")
                                    ).changed();
                                    ui.end_row();

                                    // Release
                                    ui.label("Release:");
                                    let mut release = self.settings.get_release();
                                    let release_changed = ui.add(
                                        egui::Slider::new(&mut release, 0..=2000)
                                            .text("ms")
                                    ).changed();
                                    ui.end_row();

                                    (attack_changed, attack, decay_changed, decay, 
                                     sustain_changed, sustain, release_changed, release)
                                });

                            // Update settings if any ADSR values changed
                            let (attack_changed, attack, decay_changed, decay,
                                 sustain_changed, sustain, release_changed, release) = envelope_response.inner;
                            
                            if attack_changed {
                                self.settings.set_attack(attack);
                                let _ = self.settings_sender.send(self.settings.clone());
                            }
                            if decay_changed {
                                self.settings.set_decay(decay);
                                let _ = self.settings_sender.send(self.settings.clone());
                            }
                            if sustain_changed {
                                self.settings.set_sustain(sustain);
                                let _ = self.settings_sender.send(self.settings.clone());
                            }
                            if release_changed {
                                self.settings.set_release(release);
                                let _ = self.settings_sender.send(self.settings.clone());
                            }
                        });
                    });
                });

            // Add Modulation section
            ui.group(|ui| {
                ui.heading("Modulation");
                ui.add_space(10.0);

                // Portamento control
                ui.horizontal(|ui| {
                    ui.label("Portamento:");
                    let mut portamento = self.settings.get_portamento_time();
                    if ui.add(
                        egui::Slider::new(&mut portamento, 0.0..=2.0)
                            .text("s")
                    ).changed() {
                        self.settings.set_portamento_time(portamento);
                        let _ = self.settings_sender.send(self.settings.clone());
                    }
                });

                ui.add_space(5.0);

                // LFO controls
                ui.horizontal(|ui| {
                    ui.label("LFO Frequency:");
                    let mut lfo_freq = self.settings.get_lfo_freq();
                    if ui.add(
                        egui::Slider::new(&mut lfo_freq, LFO_MIN_FREQ..=LFO_MAX_FREQ)
                            .logarithmic(true)
                            .text("Hz")
                    ).changed() {
                        self.settings.set_lfo_freq(lfo_freq);
                        let _ = self.settings_sender.send(self.settings.clone());
                    }
                });

                ui.horizontal(|ui| {
                    ui.label("LFO Amount:");
                    let mut lfo_amount = self.settings.get_lfo_amount();
                    if ui.add(
                        egui::Slider::new(&mut lfo_amount, 0.0..=1.0)
                            .text("")
                    ).changed() {
                        self.settings.set_lfo_amount(lfo_amount);
                        let _ = self.settings_sender.send(self.settings.clone());
                    }
                });
            });

            // Add Effects section
            ui.group(|ui| {
                ui.heading("Effects");
                ui.add_space(10.0);
                
                Self::render_effect_controls(ui, &self.settings.effect1, "Effect 1", &self.settings, &self.settings_sender);
                ui.add_space(5.0);
                Self::render_effect_controls(ui, &self.settings.effect2, "Effect 2", &self.settings, &self.settings_sender);
            });

            // Add Mode control
            ui.group(|ui| {
                ui.horizontal(|ui| {
                    ui.label("Mode:");
                    let mut current_mode = self.settings.play_mode.load(Ordering::Relaxed);
                    if ui.selectable_value(&mut current_mode, MODE_POLY, "Poly").clicked() {
                        self.settings.play_mode.store(current_mode, Ordering::Relaxed);
                        let _ = self.settings_sender.send(self.settings.clone());
                    }
                    if ui.selectable_value(&mut current_mode, MODE_MONO, "Mono").clicked() {
                        self.settings.play_mode.store(current_mode, Ordering::Relaxed);
                        let _ = self.settings_sender.send(self.settings.clone());
                    }
                });
            });
        });
    }
}

impl eframe::App for SynthUI {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Set up a dark theme
        ctx.set_visuals(egui::Visuals::dark());

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.vertical_centered(|ui| {
                ui.heading("WTSynth");
                ui.add_space(10.0);
            });

            // Main container with spacing
            egui::Grid::new("synth_grid")
                .spacing([20.0, 20.0])
                .show(ui, |ui| {
                    // Left column - Voices
                    ui.vertical(|ui| {
                        ui.group(|ui| {
                            ui.set_width(300.0);
                            ui.heading("Voices");
                            ui.add_space(10.0);
                            
                            let settings = &self.settings;
                            let settings_sender = &self.settings_sender;
                            
                            Self::render_voice_controls(ui, &settings.voice1, "Voice 1", settings, settings_sender);
                            ui.add_space(5.0);
                            Self::render_voice_controls(ui, &settings.voice2, "Voice 2", settings, settings_sender);
                            ui.add_space(5.0);
                            Self::render_voice_controls(ui, &settings.voice3, "Voice 3", settings, settings_sender);
                        });
                    });

                    // Right column - Filter and Envelope
                    ui.vertical(|ui| {
                        // Filter section
                        ui.group(|ui| {
                            ui.set_width(300.0);
                            ui.heading("Filter");
                            ui.add_space(10.0);
                            
                            // Filter mode selection with custom styling
                            ui.horizontal(|ui| {
                                ui.label("Mode:");
                                let mut current_mode = self.settings.get_filter_mode();
                                ui.selectable_value(&mut current_mode, FILTER_MODE_LOW, "Low Pass");
                                ui.selectable_value(&mut current_mode, FILTER_MODE_BAND, "Band Pass");
                                ui.selectable_value(&mut current_mode, FILTER_MODE_HIGH, "High Pass");
                                if current_mode != self.settings.get_filter_mode() {
                                    self.settings.set_filter_mode(current_mode);
                                    let _ = self.settings_sender.send(self.settings.clone());
                                }
                            });

                            ui.add_space(5.0);

                            // Cutoff with logarithmic slider
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

                            ui.add_space(5.0);
                            
                            // Resonance with visual feedback
                            ui.horizontal(|ui| {
                                ui.label("Resonance:");
                                let mut resonance = self.settings.get_filter_resonance();
                                if ui.add(
                                    egui::Slider::new(&mut resonance, MIN_RESONANCE..=MAX_RESONANCE)
                                        .text("")
                                ).changed() {
                                    self.settings.set_filter_resonance(resonance);
                                    let _ = self.settings_sender.send(self.settings.clone());
                                }
                            });
                        });

                        ui.add_space(10.0);

                        // ADSR Envelope section
                        ui.group(|ui| {
                            ui.set_width(300.0);
                            ui.heading("Envelope");
                            ui.add_space(10.0);

                            let envelope_response = egui::Grid::new("envelope_grid")
                                .spacing([10.0, 10.0])
                                .show(ui, |ui| {
                                    // Attack
                                    ui.label("Attack:");
                                    let mut attack = self.settings.get_attack();
                                    let attack_changed = ui.add(
                                        egui::Slider::new(&mut attack, 0..=2000)
                                            .text("ms")
                                    ).changed();
                                    ui.end_row();

                                    // Decay
                                    ui.label("Decay:");
                                    let mut decay = self.settings.get_decay();
                                    let decay_changed = ui.add(
                                        egui::Slider::new(&mut decay, 0..=2000)
                                            .text("ms")
                                    ).changed();
                                    ui.end_row();

                                    // Sustain
                                    ui.label("Sustain:");
                                    let mut sustain = self.settings.get_sustain();
                                    let sustain_changed = ui.add(
                                        egui::Slider::new(&mut sustain, 0.0..=1.0)
                                            .text("")
                                    ).changed();
                                    ui.end_row();

                                    // Release
                                    ui.label("Release:");
                                    let mut release = self.settings.get_release();
                                    let release_changed = ui.add(
                                        egui::Slider::new(&mut release, 0..=2000)
                                            .text("ms")
                                    ).changed();
                                    ui.end_row();

                                    (attack_changed, attack, decay_changed, decay, 
                                     sustain_changed, sustain, release_changed, release)
                                });

                            // Update settings if any ADSR values changed
                            let (attack_changed, attack, decay_changed, decay,
                                 sustain_changed, sustain, release_changed, release) = envelope_response.inner;
                            
                            if attack_changed {
                                self.settings.set_attack(attack);
                                let _ = self.settings_sender.send(self.settings.clone());
                            }
                            if decay_changed {
                                self.settings.set_decay(decay);
                                let _ = self.settings_sender.send(self.settings.clone());
                            }
                            if sustain_changed {
                                self.settings.set_sustain(sustain);
                                let _ = self.settings_sender.send(self.settings.clone());
                            }
                            if release_changed {
                                self.settings.set_release(release);
                                let _ = self.settings_sender.send(self.settings.clone());
                            }
                        });
                    });
                });

            // Add Modulation section
            ui.group(|ui| {
                ui.heading("Modulation");
                ui.add_space(10.0);

                // Portamento control
                ui.horizontal(|ui| {
                    ui.label("Portamento:");
                    let mut portamento = self.settings.get_portamento_time();
                    if ui.add(
                        egui::Slider::new(&mut portamento, 0.0..=2.0)
                            .text("s")
                    ).changed() {
                        self.settings.set_portamento_time(portamento);
                        let _ = self.settings_sender.send(self.settings.clone());
                    }
                });

                ui.add_space(5.0);

                // LFO controls
                ui.horizontal(|ui| {
                    ui.label("LFO Frequency:");
                    let mut lfo_freq = self.settings.get_lfo_freq();
                    if ui.add(
                        egui::Slider::new(&mut lfo_freq, LFO_MIN_FREQ..=LFO_MAX_FREQ)
                            .logarithmic(true)
                            .text("Hz")
                    ).changed() {
                        self.settings.set_lfo_freq(lfo_freq);
                        let _ = self.settings_sender.send(self.settings.clone());
                    }
                });

                ui.horizontal(|ui| {
                    ui.label("LFO Amount:");
                    let mut lfo_amount = self.settings.get_lfo_amount();
                    if ui.add(
                        egui::Slider::new(&mut lfo_amount, 0.0..=1.0)
                            .text("")
                    ).changed() {
                        self.settings.set_lfo_amount(lfo_amount);
                        let _ = self.settings_sender.send(self.settings.clone());
                    }
                });
            });

            // Add Effects section
            ui.group(|ui| {
                ui.heading("Effects");
                ui.add_space(10.0);
                
                Self::render_effect_controls(ui, &self.settings.effect1, "Effect 1", &self.settings, &self.settings_sender);
                ui.add_space(5.0);
                Self::render_effect_controls(ui, &self.settings.effect2, "Effect 2", &self.settings, &self.settings_sender);
            });

            // Add Mode control
            ui.group(|ui| {
                ui.horizontal(|ui| {
                    ui.label("Mode:");
                    let mut current_mode = self.settings.play_mode.load(Ordering::Relaxed);
                    if ui.selectable_value(&mut current_mode, MODE_POLY, "Poly").clicked() {
                        self.settings.play_mode.store(current_mode, Ordering::Relaxed);
                        let _ = self.settings_sender.send(self.settings.clone());
                    }
                    if ui.selectable_value(&mut current_mode, MODE_MONO, "Mono").clicked() {
                        self.settings.play_mode.store(current_mode, Ordering::Relaxed);
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

fn wavetable_triangle() -> Vec<f32> {
    let wave_table_size = WAVETABLE_SIZE;
    let mut wave_table: Vec<f32> = Vec::with_capacity(wave_table_size);
    for n in 0..wave_table_size {
        let phase = n as f32 / wave_table_size as f32;
        wave_table.push(if phase < 0.5 {
            4.0 * phase - 1.0
        } else {
            3.0 - 4.0 * phase
        });
    }
    wave_table
}

fn wavetable_noise() -> Vec<f32> {
    let wave_table_size = WAVETABLE_SIZE;
    let mut wave_table: Vec<f32> = Vec::with_capacity(wave_table_size);
    let mut rng = rand::thread_rng();
    for _ in 0..wave_table_size {
        wave_table.push(rng.gen_range(-1.0..=1.0));
    }
    wave_table
}

#[derive(Debug)]
pub enum SynthError {
    AudioError(String),
    MidiError(String),
    ThreadError(String),
}

impl std::error::Error for SynthError {}

impl std::fmt::Display for SynthError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SynthError::AudioError(msg) => write!(f, "Audio error: {}", msg),
            SynthError::MidiError(msg) => write!(f, "MIDI error: {}", msg),
            SynthError::ThreadError(msg) => write!(f, "Thread error: {}", msg),
        }
    }
}

impl From<rodio::StreamError> for SynthError {
    fn from(err: rodio::StreamError) -> Self {
        SynthError::AudioError(err.to_string())
    }
}

impl From<std::io::Error> for SynthError {
    fn from(err: std::io::Error) -> Self {
        SynthError::MidiError(err.to_string())
    }
}

fn wavetable_main(
    settings: SynthSettings,
    target_frequency: f32,
    velocity: f32,
    shared: Arc<Mutex<f32>>,
    wave_type: u8,
    from_frequency: Option<f32>,
) -> Result<thread::JoinHandle<()>, SynthError> {
    let note = std::thread::spawn(move || {
        let wave_table: Vec<f32> = match wave_type {
            WAVE_TYPE_SINE => wavetable_sine(),
            WAVE_TYPE_SAW => wavetable_saw(),
            WAVE_TYPE_SQUARE => wavetable_square(),
            WAVE_TYPE_TRIANGLE => wavetable_triangle(),
            WAVE_TYPE_NOISE => wavetable_noise(),
            _ => wavetable_sine(),
        };
        
        let mut oscillator = WavetableOscillator::new(SAMPLE_RATE, wave_table, Arc::clone(&shared));
        
        // Set portamento time before setting frequencies
        oscillator.portamento_time = settings.get_portamento_time();
        
        // Set the starting frequency if provided, otherwise use target frequency
        oscillator.current_frequency = from_frequency.unwrap_or(target_frequency);
        oscillator.target_frequency = target_frequency;
        oscillator.update_index_increment();
        
        oscillator.set_amplitude(velocity/127.0);
        
        oscillator.set_attack(settings.get_attack());
        oscillator.set_decay(settings.get_decay());
        oscillator.set_sustain(settings.get_sustain());
        oscillator.set_release(settings.get_release());
        
        oscillator.set_filter_cutoff(settings.get_filter_cutoff());
        oscillator.set_filter_resonance(settings.get_filter_resonance());
        oscillator.set_filter_mode(settings.get_filter_mode());

        oscillator.lfo_freq = settings.get_lfo_freq();
        oscillator.lfo_amount = settings.get_lfo_amount();
        
        // Set up LFO wavetable
        oscillator.lfo_wave_table = wavetable_sine(); // Using sine LFO for now
        
        // Set up effects
        oscillator.effect1_type = settings.effect1.effect_type.load(Ordering::Relaxed);
        oscillator.effect2_type = settings.effect2.effect_type.load(Ordering::Relaxed);

        // Initialize Distortion if needed
        if oscillator.effect1_type == EFFECT_DISTORTION {
            let drive = *settings.effect1.drive.lock().unwrap();
            let mix = *settings.effect1.mix.lock().unwrap();
            oscillator.distortion1 = Some(Distortion::new(drive, mix));
        } else if oscillator.effect2_type == EFFECT_DISTORTION {
            let drive = *settings.effect2.drive.lock().unwrap();
            let mix = *settings.effect2.mix.lock().unwrap();
            oscillator.distortion2 = Some(Distortion::new(drive, mix));
        }

        if let Ok((_stream, stream_handle)) = OutputStream::try_default() {
            if let Err(e) = stream_handle.play_raw(oscillator.convert_samples()) {
                eprintln!("Error playing audio: {}", e);
                return;
            }
            
            while shared.lock().map(|guard| *guard > 0.0).unwrap_or(false) {
                thread::sleep(Duration::from_micros(1));
            }
            thread::sleep(Duration::from_millis(settings.get_release() as u64));
        } else {
            eprintln!("Error opening audio stream");
        }
    });

    Ok(note)
}

fn main() {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([640.0, 720.0])
            .with_min_inner_size([640.0, 720.0])
            .with_title("WTSynth")
            .with_resizable(false),
        ..Default::default()
    };
    
    eframe::run_native(
        "WTSynth",
        options,
        Box::new(|cc| Box::new(SynthUI::new(cc))),
    )
    .unwrap();
}

struct VoiceManager {
    voices: HashMap<u8, NoteData>,
    settings: VoiceSettings,
    last_note: Option<u8>,
    note_stack: Vec<u8>,
}

impl VoiceManager {
    fn new(settings: VoiceSettings) -> Self {
        Self {
            voices: HashMap::with_capacity(16),
            settings,
            last_note: None,
            note_stack: Vec::new(),
        }
    }

    fn note_on(&mut self, note: u8, velocity: f32, synth_settings: &SynthSettings) -> Result<(), SynthError> {
        if !self.settings.is_enabled() {
            return Ok(());
        }

        let is_mono = synth_settings.play_mode.load(Ordering::Relaxed) == MODE_MONO;
        
        if is_mono {
            // Clear any existing voices first
            self.clear_all_voices();
            
            // In mono mode, keep track of all pressed notes
            if !self.note_stack.contains(&note) {
                self.note_stack.push(note);
            }
            
            // Only play the most recent note
            if Some(note) != self.last_note {
                // Start new note with portamento from last frequency
                let from_freq = self.last_note.map(|n| self.calculate_frequency(n));
                self.start_note(note, velocity, synth_settings, from_freq)?;
                self.last_note = Some(note);
            }
        } else {
            // Check voice limit in poly mode
            let max_voices = *synth_settings.max_polyphony.lock().unwrap();
            if self.voices.len() >= max_voices as usize {
                // Find and stop the oldest note
                if let Some(&oldest_note) = self.voices.keys().next() {
                    self.note_off(oldest_note, synth_settings)?;
                }
            }
            self.start_note(note, velocity, synth_settings, None)?;
        }
        
        Ok(())
    }

    fn note_off(&mut self, note: u8, synth_settings: &SynthSettings) -> Result<(), SynthError> {
        if synth_settings.play_mode.load(Ordering::Relaxed) == MODE_MONO {
            // Remove note from stack
            if let Some(pos) = self.note_stack.iter().position(|&n| n == note) {
                self.note_stack.remove(pos);
            }
            
            // If there are other held notes, switch to the last one
            if let Some(&last_held) = self.note_stack.last() {
                if Some(note) == self.last_note {
                    // Use a default velocity of 64 for note retriggering
                    self.note_on(last_held, 64.0, synth_settings)?;
                }
            } else {
                // No more held notes
                self.last_note = None;
            }
        }
        
        // Stop the note
        if let Some(note_data) = self.voices.remove(&note) {
            if let Ok(mut note_shared_vel) = note_data.shared.lock() {
                *note_shared_vel = 0.0;
            }
        }
        Ok(())
    }

    fn start_note(&mut self, note: u8, velocity: f32, synth_settings: &SynthSettings, 
                 from_freq: Option<f32>) -> Result<(), SynthError> {
        let shared = Arc::new(Mutex::new(1.0));
        let target_freq = self.calculate_frequency(note);
        let adjusted_velocity = velocity * self.settings.get_volume();

        let note_handle = wavetable_main(
            synth_settings.clone(),
            target_freq,
            adjusted_velocity,
            Arc::clone(&shared),
            self.settings.get_wave_type(),
            from_freq,
        )?;

        self.voices.insert(note, NoteData::new(note_handle, shared));
        Ok(())
    }

    fn calculate_frequency(&self, note: u8) -> f32 {
        let semitones = self.settings.get_semitones() as f32;
        let cents = self.settings.get_cents() / 100.0;
        let total_semitones = semitones + cents;
        440.0 * 2_f32.powf((note as f32 - 69.0 + total_semitones) / 12.0)
    }

    fn clear_all_voices(&mut self) {
        for (_, note_data) in self.voices.drain() {
            if let Ok(mut note_shared_vel) = note_data.shared.lock() {
                *note_shared_vel = 0.0;
            }
        }
    }
}

fn run(settings_receiver: mpsc::Receiver<SynthSettings>) -> Result<(), Box<dyn Error>> {
    let mut input = String::new();
    let mut voice_managers = vec![
        VoiceManager::new(SynthSettings::default().voice1),
        VoiceManager::new(SynthSettings::default().voice2),
        VoiceManager::new(SynthSettings::default().voice3),
    ];

    let mut midi_in = MidiInput::new("midi_read_fx")?;
    midi_in.ignore(Ignore::None);

    // Get an input port (read from console if multiple are available)
    let in_ports = midi_in.ports();
    let in_port = match in_ports.len() {
        0 => return Err("no input port found".into()),
        1 => {
            println!(
                "Choosing the only available input port: {}",
                midi_in.port_name(&in_ports[0])?
            );
            &in_ports[0]
        }
        _ => {
            println!("\nAvailable input ports:");
            for (i, p) in in_ports.iter().enumerate() {
                println!("{}: {}", i, midi_in.port_name(p)?);
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
                current_settings = new_settings.clone();
                for (i, manager) in voice_managers.iter_mut().enumerate() {
                    manager.settings = match i {
                        0 => current_settings.voice1.clone(),
                        1 => current_settings.voice2.clone(),
                        2 => current_settings.voice3.clone(),
                        _ => unreachable!(),
                    };
                }
            }

            match message.len() {
                3 => {
                    if message[0] == 0x80 || (message[0] == 0x90 && message[2] == 0) {
                        // Note off
                        for manager in voice_managers.iter_mut() {
                            if let Err(e) = manager.note_off(message[1], &current_settings) {
                                eprintln!("Error stopping note: {}", e);
                            }
                        }
                    } else if message[0] == 0x90 {
                        // Note on
                        for manager in voice_managers.iter_mut() {
                            if let Err(e) = manager.note_on(message[1], message[2] as f32, &current_settings) {
                                eprintln!("Error starting note: {}", e);
                            }
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

#[derive(Clone)]
struct Distortion {
    drive: f32,
    mix: f32,
}

impl Distortion {
    fn new(drive: f32, mix: f32) -> Self {
        Self { drive, mix }
    }

    fn process(&self, input: f32) -> f32 {
        let distorted = input * self.drive;
        let clipped = distorted.tanh();
        input * (1.0 - self.mix) + clipped * self.mix
    }
}