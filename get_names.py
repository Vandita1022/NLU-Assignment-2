import random
from collections import Counter

# ---------- STEP 1: LOAD ----------
with open("TrainingNames.txt", "r", encoding="utf-8") as f:
    names = [line.strip() for line in f if line.strip()]

print("Original count:", len(names))


# ---------- STEP 2: REMOVE DUPLICATES ----------
names = list(set(names))
print("After removing duplicates:", len(names))


# ---------- STEP 3: LETTER COUNT ----------
letters = [name[0].upper() for name in names]
letter_counts = Counter(letters)


# ---------- STEP 4: ADD NAMES IF < 1000 ----------

needed = 1000 - len(names)

if needed > 0:
    print(f"Adding {needed} new names...")

    # Backup pool (clean Indian names not used before)
    extra_names = [
        "Abeer", "Aadit", "Aarjit", "Aaryav", "Ainesh",
        "Zeeshan", "Zidan", "Zivaan", "Zoravar",
        "Yuvan", "Yatin", "Yashit",
        "Waris", "Wajid",
        "Udarsh", "Ujith",
        "Qadir", "Qasim",
        "Omesh", "Oviya"
    ]

    # Remove already existing names
    extra_names = [n for n in extra_names if n not in names]

    # Sort letters by lowest frequency first
    sorted_letters = sorted(letter_counts.items(), key=lambda x: x[1])

    i = 0
    while needed > 0 and i < len(extra_names):
        name = extra_names[i]

        if name not in names:
            names.append(name)
            needed -= 1

        i += 1

    print("After adding:", len(names))


# ---------- STEP 5: FINAL TRIM IF NEEDED ----------
if len(names) > 1000:
    names = random.sample(names, 1000)


# ---------- STEP 6: SHUFFLE ----------
for _ in range(5):
    random.shuffle(names)


# ---------- STEP 7: SAVE ----------
with open("cleaned_names.txt", "w", encoding="utf-8") as f:
    for name in names:
        f.write(name + "\n")


print("Final count:", len(names))