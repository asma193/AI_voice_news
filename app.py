import streamlit as st
from transformers import pipeline

# 🎨 Page setup
st.set_page_config(page_title="📚 AI Story Generator", page_icon="✨")
st.title("📚 AI 500-Word Story Generator with Summary 🎬")
st.write("Choose or enter a story idea — and AI will write a story, then summarize it like a movie plot!")

# 🎯 Story ideas dropdown
st.write("💡 Choose a story idea or write your own:")
story_options = [
    "The Robot Who Learned to Love",
    "Lost Astronaut: A Journey Beyond the Stars",
    "The Hidden Kingdom Beneath the Ocean",
    "A Message from the Future",
    "The Magic Lamp That Granted One Wish Too Many",
    "When Earth Took Its Last Breath",
    "The Hacker Who Saved Humanity",
    "Whispers of the Wild: A Talking Tiger’s Tale",
    "Reflections of Another World",
    "The Forgotten Spell of the Ancient Wizard"
]
topic = st.selectbox("🎯 Select Story Topic:", story_options)
custom_topic = st.text_input("✍️ Or write your own idea here:")
if custom_topic.strip():
    topic = custom_topic

# 🧠 Load models once
@st.cache_resource
def load_models():
    story_gen = pipeline("text-generation", model="gpt2")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    return story_gen, summarizer

story_gen, summarizer = load_models()

# ✨ Generate story button
if st.button("🚀 Generate My Story"):
    if topic.strip():
        with st.spinner("🧠 Writing your story... please wait..."):
            # Generate the 500-word story
            prompt = f"Write a creative, detailed, and emotional 500-word story about: {topic}."
            story = story_gen(prompt, max_length=700, temperature=0.9, do_sample=True)[0]['generated_text']

            st.subheader("📖 AI-Generated Story:")
            st.write(story)

        with st.spinner("🎬 Creating short summary..."):
            # Summarize the generated story
            summary = summarizer(story, max_length=120, min_length100se)[0]['summary_text']
            st.subheader("🎞️ Story Summary:")
            st.success(summary)
    else:
        st.warning("⚠️ Please select or enter a topic first!")

st.caption("Built with ❤️ using Python, Streamlit & HuggingFace Transformers")
