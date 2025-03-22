from predict import predict_message

print("🧠 Fake News Credibility Checker")
print("Type a message and get its credibility score.")
print("Type 'quit' to exit.\n")

while True:
    user_input = input("Enter a message: ")
    if user_input.lower() == "quit":
        break

    score = predict_message(user_input)
    print(f"🟢 Credibility score: {score:.2f}%")

    # Response by score
    if score > 80:
        print("✅ This message seems highly reliable.\n")
    elif score > 50:
        print("⚠️ This message might be credible, but further verification is recommended.\n")
    elif score > 20:
        print("❗ This message seems suspicious. Be cautious.\n")
    else:
        print("🚫 This message is likely fake news.\n")
