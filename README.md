# Personalized Migraine Relief Through Binaural Beats

Ever wondered if your brain waves could guide your own migraine treatment? This project does exactly that. We analyze high-density EEG recordings to understand what's happening in your brain, then create custom audio therapy specifically tuned to your unique brain patterns.

Think of it like this: instead of giving everyone the same generic relaxation audio, we're reading your brain's signals and crafting therapeutic sounds that match what *your* brain needs.

## What This Does

This system takes 128-channel EEG recordings from migraine patients and:
1. **Figures out your migraine type** - Are you experiencing auras? No auras? Or are you a healthy control? Our Random Forest classifier gets it right 84.6% of the time.
2. **Analyzes your brain patterns** - We extract 1,738 different features from your EEG data to understand what's unique about your brain activity.
3. **Creates personalized audio** - Based on your specific brain patterns, age, gender, and migraine type, we generate therapeutic binaural beats tuned precisely to what you need.
4. **Generates a detailed report** - You get a full explanation of why your therapy was designed the way it was.

## The Data We're Working With

We're using a pretty remarkable dataset - 31 patients who volunteered for high-density EEG recordings:
- 18 healthy controls (never had migraines)
- 9 patients with migraine aura (those visual disturbances before the headache)
- 4 patients with non-aura migraines

Each person sat for about **13 minutes** of resting-state recording with **128 electrodes** placed on their scalp, sampling at 512 Hz. That gives us incredibly detailed brain activity data - think of it as taking 512 snapshots every single second from 128 different brain locations.

**Fun fact**: This dataset comes from a study by Chamanzar et al. (2020) looking at cortical coherence patterns in migraine patients. They used ultra-high-density EEG because migraine isn't just about pain - it's about how different parts of your brain communicate with each other.

## Getting Started

Want to try it out? Here's how:

### Installation

```bash
cd /Users/mahmudulmashrafe/Programming/FYDP/3
pip install -r requirements.txt
```

You'll need Python with libraries like MNE (for EEG processing), scikit-learn (for machine learning), and scipy (for audio generation).

### Generate Therapy for One Patient

```bash
python3 src/main_pipeline.py --patient M1_1 --duration 600
```

This processes patient M1_1 and creates a 10-minute therapeutic audio file. In about 15 seconds, you'll get:
- `output/M1_1_binaural_beat.wav` - Your personalized audio therapy
- `output/M1_1_treatment_report.txt` - A detailed explanation of how we designed your therapy

### Process Multiple Patients at Once

```bash
python3 src/main_pipeline.py --batch --duration 300
```

This will batch-process several patients, creating 5-minute audio files for each.

### Interactive Exploration

If you prefer a more visual, step-by-step walkthrough:
```bash
jupyter notebook migraine_binaural_treatment.ipynb
```

This notebook shows you the whole pipeline with plots and explanations along the way.

## Project Organization

Here's what's in the box:

```
.
├── src/
│   ├── data_loader.py              # Loads patient data & EEG files
│   ├── feature_extraction.py       # Extracts 1,738 features from brain waves
│   ├── dataset_builder.py          # Compiles everything into a dataset
│   ├── classifier.py               # Trains the migraine classifier
│   ├── binaural_beat_generator.py  # Creates the therapeutic audio
│   └── main_pipeline.py            # Ties everything together
├── data/
│   └── dataset_resting.pkl         # Pre-processed dataset (saved for speed)
├── models/
│   └── migraine_classifier.pkl     # Pre-trained classifier
├── output/                         # Your generated audio files & reports live here
├── requirements.txt                # All the Python libraries you need
├── migraine_binaural_treatment.ipynb  # Interactive demo notebook
└── README.md                       # You are here!
```

## How It Actually Works

### Step 1: Loading Your Data
First, we grab your clinical info (age, gender, migraine type) from Excel and load your 128-channel EEG recording from a .bdf file using MNE, a powerful library for neurophysiology data.

### Step 2: Feature Extraction (Where the Magic Happens)
From your ~13 minutes of brain activity, we extract **1,738 unique features**:

- **Power Spectral Density (640 features)**: How strong are different frequency bands in each of the 128 channels?
  - Delta (0.5-4 Hz): Deep sleep waves
  - Theta (4-8 Hz): Drowsy, meditative states
  - Alpha (8-13 Hz): Relaxed but alert
  - Beta (13-30 Hz): Active thinking, sometimes anxiety
  - Gamma (30-50 Hz): High-level cognition

- **Statistical Measures (512 features)**: For each channel, we calculate mean, variance, skewness, and kurtosis - basically capturing the "personality" of each brain region.

- **Connectivity Analysis**: Using coherence, we measure how well different brain regions talk to each other. Migraine often disrupts these conversations.

- **Band Ratios**: Things like Theta/Alpha ratio can reveal imbalances that guide our therapy.

### Step 3: Classification
We use a **Random Forest classifier** (200 decision trees working together) with some clever preprocessing:
- Impute any missing values
- Standardize features (so age doesn't overshadow tiny EEG values)
- Apply PCA to reduce to 30 key components
- Use SMOTE to balance our classes (since we have way more controls than non-aura patients)

Result: **84.6% accuracy** in telling apart control vs. aura vs. non-aura patients.

### Step 4: Personalized Frequency Calculation

This is where we get mathematical. The frequency of your binaural beat isn't random - it's calculated using a precise formula based on YOUR brain's specific patterns.

## The Math Behind Your Therapy

### What's a Binaural Beat Anyway?

When you hear slightly different frequencies in each ear (say, 200 Hz in left, 207 Hz in right), your brain perceives a "beating" at the difference frequency (7 Hz). This phenomenon can entrain your brainwaves to match that frequency - it's like your brain trying to sync up with the rhythm it perceives.

The core audio generation follows:
```
Left ear:  L(t) = sin(2π × f_carrier × t)
Right ear: R(t) = sin(2π × (f_carrier + f_beat) × t)

Where:
  f_carrier = base frequency (typically 150-200 Hz)
  f_beat    = therapeutic frequency (4-12 Hz)
  t         = time in seconds
```

### Personalization Formula

Your therapeutic beat frequency isn't just pulled from a hat. Here's the actual formula we use:

```
f_beat = BASE + EEG_adjustments + Demographics_adjustments

Where:
  BASE = Starting frequency based on migraine type
         - Aura → 10 Hz (alpha band, calming hyperexcitability)
         - Non-Aura → 7 Hz (theta-alpha transition, deep relaxation)
         - Control → 10 Hz (general wellness)

  EEG_adjustments = Band-specific corrections
         - High delta (>35%) → +2 Hz (push toward alertness)
         - High theta (>30%) → +1 Hz (lift toward alpha)
         - High beta (>30%)  → -2 Hz (calm down toward theta)

  Demographics_adjustments = 
         - Age < 25: +0.5 Hz (younger brains respond to higher frequencies)
         - Age > 40: -0.5 Hz (adjust for age-related changes)
         - Female: +0.3 Hz (subtle gender difference in alpha response)
```

**Final constraint**: We clip the result to therapeutic range: **4-12 Hz**

### Advanced Personalization (Precision Mode)

For even more precise targeting, we can use a continuous formula that doesn't just use thresholds:

```
f_beat = BASE + (α_deficit × W_α) + (θ_excess × W_θ) + (δ_excess × W_δ)

Where:
  α_deficit = max(0, 20% - measured_alpha)
  θ_excess  = max(0, measured_theta - 20%)
  δ_excess  = max(0, measured_delta - 15%)
  
  W_α = 0.10  (weight for alpha deficit)
  W_θ = 0.05  (weight for theta excess)
  W_δ = 0.03  (weight for delta excess)
```

**Real example**: If your occipital alpha is only 12% (should be ~20%), and your temporal theta is 32% (high!):
```
α_deficit = 20 - 12 = 8%
θ_excess  = 32 - 20 = 12%

f_beat = 10.0 + (8 × 0.10) + (12 × 0.05)
       = 10.0 + 0.8 + 0.6
       = 11.4 Hz
```

So you'd get an 11.4 Hz beat - not a generic "10-12 Hz range," but an exact frequency calculated from your specific brain pattern.

### Step 5: Audio Generation

Once we have your frequency, we:
1. Generate two pure sine waves at the calculated frequencies
2. Apply smooth **5-second fade in/out** to avoid jarring clicks
3. Normalize the audio to prevent clipping (keeps it at 80% of max volume)
4. Save as a **44.1 kHz stereo WAV file**

The result: A therapeutic audio file that's specifically tuned to what your brain needs.

### Why These Specific Adjustments?

- **Delta correction**: High slow-wave activity often indicates sleepiness or certain types of cortical abnormalities. Pushing up toward alpha promotes relaxed alertness.

- **Theta elevation**: Excess theta is common in migraine patients, especially during interictal periods. Gentle upward nudging helps restore balance.

- **Beta reduction**: High beta can indicate anxiety or hyperarousal - both migraine triggers. Bringing it down toward theta-alpha promotes calmness.

- **Age/Gender factors**: These are subtle but research-backed. Younger brains tend to have faster dominant frequencies, and there are documented gender differences in alpha peak frequency.

## How Well Does It Perform?

Let's be honest about the numbers:

- **Cross-Validation**: 84.6% ± 6.3% accuracy across multiple splits of our training data. That's pretty solid for a 3-class medical classification problem!
  
- **Test Set**: 62.5% accuracy on the held-out test set (n=8). Why the drop? Small test set + class imbalance = higher variance. With only 8 test patients, one or two misclassifications significantly impact the percentage. This is a limitation we're open about.

- **Speed**: About 15 seconds per patient from raw EEG to therapeutic audio. Fast enough for clinical use.

- **Audio Quality**: 44.1 kHz stereo with smooth 5-second fade transitions. No harsh clicks or artifacts.

**Bottom line**: The classifier is good enough to be useful, but it's not perfect. The real strength is the personalization - even if classification isn't 100%, we're still using your individual EEG patterns to customize the therapy.

## Real Patient Examples

### Patient M3_2 - Non-Aura Migraine

Here's what happened when we processed this patient:
- **Prediction**: Non-Aura migraine (62.3% confidence)
- **EEG Finding**: Occipital alpha power was through the roof at 99.83%! That's unusually high.
- **Our Therapy**: 7.8 Hz beat frequency (theta band)
- **Reasoning**: With such high alpha already, we targeted theta-alpha transition to promote deep relaxation rather than more alpha stimulation.
- **Files Generated**: `M3_2_binaural_beat.wav` and a full treatment report

### Control Group Batch Run

When we processed 5 healthy controls, look at how personalized each therapy became:

| Patient | Age | Gender | Beat Frequency | EEG Pattern | Reasoning |
|---------|-----|--------|----------------|-------------|-----------|
| C1 | 20 | F | 10.8 Hz | Normal | Young + female → slightly higher alpha |
| C10 | 43 | M | 9.5 Hz | Normal | Older age → slightly lower frequency |
| C11 | 24 | F | 12.0 Hz | High Delta | Young + high slow waves → push to upper alpha |
| C13 | 20 | M | 11.5 Hz | Normal | Young → base+0.5 Hz |
| C14 | 20 | F | 8.8 Hz | High Beta | High anxiety marker → reduce toward theta |

Notice how even among healthy controls of similar age, the frequencies varied from 8.8 Hz to 12.0 Hz based on their individual brain patterns. That's real personalization!

## Important Things You Should Know

### Let's Talk About Clinical Reality

**This is experimental research.** We need to be crystal clear about that. While binaural beats are fascinating and there's research supporting their effects on brainwave entrainment, this specific implementation:
- Hasn't been through clinical trials
- Shouldn't replace your actual migraine medication
- Is designed as a **complementary approach**, not a cure
- Uses frequency selections based on neuroscience literature, but we haven't empirically validated the outcomes in a controlled study

**In other words**: Think of this as a high-tech relaxation tool informed by your brain patterns, not a medical device. Always talk to your doctor about migraine management.

### Dataset Reality Check

Our dataset has 31 patients. That's... not huge:
- **Class imbalance**: 18 controls, 9 with aura, only 4 non-aura patients
- **Limited generalization**: The model learned from these 31 people. Your mileage may vary.
- **Test variance**: With such a small test set, accuracy metrics bounce around a lot

This is why we're transparent about the 62.5% test accuracy - it's not bad, but it reflects our limited sample size. With more data, we'd likely see more stable performance.

## Where This Could Go Next

We've got big ideas for where to take this project:

### On the Technical Side

**Richer Data Sources**
- Right now we only use resting-state EEG. But our dataset has SSAEP (auditory stimulation) and SSVEP (visual stimulation) recordings too. Those could reveal how patients' brains respond to stimulation - super relevant for audio therapy!

**Deep Learning**
- Random Forests are great, but CNNs or RNNs could potentially learn patterns directly from raw EEG without manual feature engineering. Imagine feeding in the raw 128-channel signal and letting the network figure out what matters.

**Real-Time Processing**
- With consumer EEG headsets (like Muse or OpenBCI), we could process data on-the-fly and adapt the therapy in real-time as your brain responds. Dynamic, responsive treatment instead of pre-generated audio.

### On the Clinical Side

**Scale Up**
- We need data from 100+ patients to really validate this approach. Small sample sizes are the enemy of reliable medical AI.

**Longitudinal Studies**
- Does this actually reduce migraine frequency or severity over weeks/months? We need controlled trials with placebo groups and proper outcome measures.

**Collaborate with Neurologists**
- Getting feedback from migraine specialists would help refine the frequency selection algorithms and validate clinical utility.

### Making It Accessible

**Mobile App**
- Imagine an app where you could upload your EEG data (or connect a consumer headset) and get your personalized therapy instantly. With explanations, tracking, and progress monitoring.

**Continuous Monitoring**
- Wearable EEG that monitors your brain state and adjusts therapy throughout the day based on current patterns and migraine risk.

**Integration with Migraine Trackers**
- Connect with apps like Migraine Buddy to correlate therapy usage with actual migraine outcomes. Real-world effectiveness data!

We're dreaming big, but starting small. This FYDP project is proof of concept - the foundation for something potentially much bigger.

## Want to Learn More?

### The Science Behind This

**Our Dataset**
- Chamanzar, A., et al. (2020). "Abnormalities in cortical pattern of coherence in migraine detected using ultra high-density EEG." *Brain Communications*. 
  - These researchers collected the amazing 128-channel EEG data we're using. Their focus was on coherence patterns (basically, how synchronized different brain regions are), which is fascinating in migraine research.

**Binaural Beats Background**
- Our frequency selections are based on decades of brainwave entrainment research showing that:
  - Theta (4-8 Hz) promotes deep relaxation and meditation
  - Alpha (8-13 Hz) is associated with calm alertness and reduced cortical excitability
  - These frequencies may help modulate the cortical hyperexcitability seen in migraine patients

**The Math**
- The personalization equations we developed combine neuroscience principles with machine learning insights from the EEG patterns we observed across patient groups.

### Questions? Ideas? Want to Collaborate?

This project was developed for FYDP (Final Year Design Project) in January 2026. If you're interested in migraine research, EEG analysis, or therapeutic audio, feel free to reach out!

---

**Built with curiosity, science, and a lot of signal processing** 🧠🎧

*Remember: Your brain is unique. Your therapy should be too.*
