import streamlit as st
import requests

st.set_page_config(page_title="Math Hint Chatbot", page_icon="🧮", layout="centered")

st.title("🧮 Math Hint Chatbot")
st.caption("Class 6-10 | Hints milenge, direct answer nahi! 😊")

# Session initialize karo
if "session_id" not in st.session_state:
    st.session_state.session_id = "student_1"
if "messages" not in st.session_state:
    st.session_state.messages = []
if "hints_available" not in st.session_state:
    st.session_state.hints_available = 0
if "hint_level" not in st.session_state:
    st.session_state.hint_level = 1

# Chat history dikhao
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Question input
question = st.chat_input("Apna Math question yahan likho...")

if question:
    # User message dikhao
    with st.chat_message("user"):
        st.write(question)
    st.session_state.messages.append({"role": "user", "content": question})

    # Reset hint level for new question
    st.session_state.hint_level = 1

    # Backend se hint lo
    try:
        response = requests.post(
            "http://localhost:5000/ask",
            json={
                "question": question,
                "session_id": st.session_state.session_id,
                "hint_level": 1
            }
        )
        data = response.json()

        if data["status"] == "success":
            hint_text = "\n\n".join(data["hints"])
            confidence = data.get("confidence", "")
            total = data.get("total_hints_available", 1)
            st.session_state.hints_available = total

            reply = f"{hint_text}\n\n*Confidence: {confidence} | Total hints: {total}*"

            with st.chat_message("assistant"):
                st.write(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})

        else:
            with st.chat_message("assistant"):
                st.write(data.get("message", "Kuch samajh nahi aaya, dobara puchho!"))

    except Exception as e:
        with st.chat_message("assistant"):
            st.write(f"❌ Backend se connect nahi ho paya. Flask server chalu hai kya?")

# Aur hint button
if st.session_state.hints_available > 1:
    if st.button("💡 Aur hint chahiye"):
        try:
            response = requests.post(
                "http://localhost:5000/next-hint",
                json={"session_id": st.session_state.session_id}
            )
            data = response.json()

            if data["status"] == "success":
                hints = data["hints"]
                latest_hint = hints[-1]

                with st.chat_message("assistant"):
                    st.write(f"🔍 Next Hint:\n\n{latest_hint}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"🔍 Next Hint:\n\n{latest_hint}"
                })

            elif data["status"] == "max_hints_reached":
                with st.chat_message("assistant"):
                    st.write(data["message"])

        except:
            st.error("Backend se connect nahi ho paya!")

# Clear button
if st.button("🗑️ Clear Chat"):
    st.session_state.messages = []
    st.session_state.hints_available = 0
    st.rerun()

with st.sidebar:
    st.header("📚 Sample Questions")
    st.write("Yeh questions try kar sakte ho:")
    
    sample_questions = [
        "area of triangle",
        "pythagoras theorem", 
        "quadratic equation",
        "perimeter of circle",
        "linear equations",
        "simple interest",
        "probability",
        "volume of cylinder",
        "arithmetic progression",
        "percentage"
    ]
    
    for q in sample_questions:
        st.code(q)
    
    st.divider()
    st.warning("⚠️ Sirf Math questions puchho!")
    st.info("💡 Confidence 45% se kam ho to question aur clear likho")