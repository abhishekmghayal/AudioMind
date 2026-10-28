from services.audio_pipeline import AudioPipeline

segs = [{"start": 2.0, "end": 8.0}, {"start": 12.0, "end": 20.0}]
gaps = AudioPipeline.find_gaps(segs, 25.0)
assert gaps == [(0.0, 1.8), (8.2, 11.8), (20.2, 25.0)], gaps
assert AudioPipeline.find_gaps([{"start": 0.0, "end": 10.0}], 10.0) == []
print("PASS: gaps")