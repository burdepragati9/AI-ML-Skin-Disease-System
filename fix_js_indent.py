path = r"c:\Updated_Project\frontend\src\pages\prediction\ImagePrediction.jsx"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

bad = "const detectRes = await api.post('/predict/detect-face', detectFormData);"
good = "        const detectRes = await api.post('/predict/detect-face', detectFormData);"

# Replace all occurrences of the unindented line (only the one at column 0)
count = content.count("\n" + bad + "\n")
if count:
    content = content.replace("\n" + bad + "\n", "\n" + good + "\n", 1)
    print("Fixed", count, "occurrence(s) of unindented detectRes line")
else:
    print("No unindented line found (already correct or different whitespace)")

with open(path, "w", encoding="utf-8") as f:
    f.write(content)
print("Done")
