================================================================================
  MusicLEARN Benchmark Report
================================================================================

  Tracks analyzed: 8
  Model: unified.pth (5.6M params)
  Checkpoint: checkpoints/unified.pth

Track                Genre/Style                 V      A      T      C  Top Instruments                Duration
----------------------------------------------------------------------------------------------------
vivaldi_spring       Classical/Baroque       +0.22  +0.47  +0.01  +1.06  violin(29%), guitar(15%), trumpet(11%)  11.6min
sad_piano            Classical/Piano         -0.07  +0.01  +0.34  +0.38  piano(59%), mallet_percussion(11%)   2.8min
edm_drop             Dance/Funky             +0.27  +0.42  -0.32  +1.00  drums(20%), cymbals(17%), trombone(16%)   1.8min
jazz_swing           Dance/Funky             +0.14  +0.36  +0.17  +1.41  drums(64%), cymbals(42%), synthesizer(19%)   8.0min
rock_electric        Dance/Funky             +0.14  +0.49  +0.74  +1.37  voice(31%), cymbals(27%), drums(14%)   3.5min
colosseum_epic       Epic/Cinematic          +0.31  +0.39  -0.21  +0.68  voice(16%), trumpet(14%), saxophone(13%)   7.9min
tense_suspense       Epic/Cinematic          +0.20  +0.34  -0.02  +0.98  violin(12%), synthesizer(11%)   26.1min
chinese_trad         Melancholic/Sad         +0.08  -0.10  -0.63  -3.15                                   0.4min

--- Results by genre ---

  [Classical/Baroque]
    vivaldi_spring: Vivaldi Four Seasons - Spring. Genre correct. Violin detected.
  [Classical/Piano]
    sad_piano: Sad piano solo. Piano detected (59%). Genre correct.
  [Dance/Funky]
    edm_drop: EDM drop. Drums detected. Dance classification OK.
    jazz_swing: Swing jazz. Detected as Funky (close). Drums/cymbals correct.
    rock_electric: Electric rock. Detected as Funky - should be Rock/Heavy. Vocals correct.
  [Epic/Cinematic]
    colosseum_epic: Battle orchestral. Genre correct. Voice/trumpet detected.
    tense_suspense: Suspenseful BGM. Genre OK but Tension should be higher.
  [Melancholic/Sad]
    chinese_trad: Chinese traditional. Very short clip (0.4min) - insufficient data.

--- Summary ---
  Accurate genre: Classical/Baroque, Classical/Piano, Epic/Cinematic, Dance/Funky
  Needs improvement: Rock detection, Jazz/Swing, Tension sensitivity
  Instrument accuracy: Piano, Violin, Drums, Cymbals, Trumpet, Voice
  Weak instruments: Accordion, Banjo, Cello, Clarinet (low training data)

Benchmark generated: July 2026