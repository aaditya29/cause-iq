import json

with open('../data/raw/multiwoz_2.2/train/dialogues_001.json', 'r') as f:
    data = json.load(f)

# Check first conversation
first_conv = data[0]
print(f"Conversation ID: {first_conv.get('dialogue_id')}")
print(f"Services: {first_conv.get('services')}")

# Check the actual structure
turns = first_conv.get('turns', {})
print(f"\nTurns structure:")
print(f"Type: {type(turns)}")
print(f"Keys: {turns.keys()}")
if 'turn_id' in turns:
    print(f"\nNumber of turns: {len(turns['turn_id'])}")
    print(f"First turn_id: {turns['turn_id'][0]}")
    print(f"First speaker: {turns['speaker'][0]}")
    print(f"First utterance: {turns['utterance'][0][:100]}...")

    # Check frames structure
    if 'frames' in turns and turns['frames']:
        print(f"\nFirst turn frames type: {type(turns['frames'][0])}")
        if isinstance(turns['frames'][0], list):
            print(f"Number of frames in first turn: {len(turns['frames'][0])}")
            if turns['frames'][0]:
                print(
                    f"First frame keys: {turns['frames'][0][0].keys() if turns['frames'][0] else 'empty'}")

first_conv = data[0]
turns = first_conv['turns']

print("Detailed structure check:")
print(f"turn_id: {turns['turn_id'][:3]}")
print(f"speaker: {turns['speaker'][:3]}")
print(f"utterance: {turns['utterance'][:2]}")
print(f"\nframes[0] type: {type(turns['frames'][0])}")
print(
    f"frames[0] keys: {turns['frames'][0].keys() if isinstance(turns['frames'][0], dict) else 'not a dict'}")
print(f"\ndialogue_acts[0] type: {type(turns['dialogue_acts'][0])}")
print(f"dialogue_acts[0]: {turns['dialogue_acts'][0]}")
