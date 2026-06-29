from flask import Flask, request, jsonify
from flask_cors import CORS
from hint_engine import HintEngine

app = Flask(__name__)
CORS(app)

print("🚀 Loading Hint Engine...")
engine = HintEngine(csv_path='data/maths_only.csv')
print("✅ Engine ready!")

session_hints = {}

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "status": "running",
        "message": "Math Hint Chatbot Backend is live!"
    })

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.get_json()

    if not data or 'question' not in data:
        return jsonify({"error": "Question field required"}), 400

    user_question = data['question'].strip()
    session_id = data.get('session_id', 'default')
    hint_level = data.get('hint_level', 1)

    if not user_question:
        return jsonify({"error": "Question cannot be empty"}), 400

    result = engine.get_hint(user_question)

    if not result['found']:
        return jsonify({
            "status": "not_found",
            "message": result['message'],
            "hints": [],
            "confidence": "0%"
        })

    session_hints[session_id] = {
        "question": user_question,
        "all_hints": result['hints'],
        "current_level": hint_level
    }

    max_level = min(hint_level, len(result['hints']))
    hints_to_show = result['hints'][:max_level]

    return jsonify({
        "status": "success",
        "question_understood": result['matched_question'],
        "topic": result['topic'],
        "hints": hints_to_show,
        "total_hints_available": len(result['hints']),
        "hint_level": max_level,
        "confidence": f"{round(result['score'] * 100, 1)}%",
        "message": "Direct answer nahi dunga — hints se try karo! 😊"
    })

@app.route('/next-hint', methods=['POST'])
def next_hint():
    data = request.get_json()
    session_id = data.get('session_id', 'default')

    if session_id not in session_hints:
        return jsonify({"error": "Pehle koi question puchho!"}), 400

    session = session_hints[session_id]
    current_level = session['current_level']
    all_hints = session['all_hints']

    if current_level >= len(all_hints):
        return jsonify({
            "status": "max_hints_reached",
            "message": "Maine sari hints de di hain! Ab khud try karo — tum kar sakte ho! 💪",
            "hints": all_hints
        })

    next_level = current_level + 1
    session_hints[session_id]['current_level'] = next_level

    return jsonify({
        "status": "success",
        "hints": all_hints[:next_level],
        "hint_level": next_level,
        "total_hints_available": len(all_hints)
    })

@app.route('/topics', methods=['GET'])
def get_topics():
    topics = engine.df['context'].dropna().unique().tolist()
    return jsonify({"topics": topics})

if __name__ == '__main__':
    app.run(debug=True, port=5000)