import pickle
import json

# Load the pickle files
with open('words.pkl', 'rb') as f:
    words = pickle.load(f)

with open('classes.pkl', 'rb') as f:
    classes = pickle.load(f)

# Save them as JSON files
with open('words.json', 'w') as f:
    json.dump(words, f)

with open('classes.json', 'w') as f:
    json.dump(classes, f)

print("Conversion to JSON complete!")
