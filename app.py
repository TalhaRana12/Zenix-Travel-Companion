import os
import time
import json
import gradio as gr
from groq import Groq, APIError
from urllib.parse import quote_plus

# --- DATABASE CONFIGURATION ---
FEEDBACK_DB_FILE = "travel_feedback_db.json"

# --- DATA STORAGE (FILE-BASED DATABASE) ---
def load_feedback_database():
    """Load feedback data from JSON file database."""
    try:
        if os.path.exists(FEEDBACK_DB_FILE):
            with open(FEEDBACK_DB_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            # Initialize with empty database structure
            initial_data = {
                "feedback_entries": [],
                "metadata": {
                    "total_feedback": 0,
                    "average_rating": 0,
                    "last_updated": time.strftime("%Y-%m-%d %H:%M:%S")
                }
            }
            save_feedback_database(initial_data)
            return initial_data
    except Exception as e:
        print(f"Error loading database: {e}")
        return {"feedback_entries": [], "metadata": {"total_feedback": 0, "average_rating": 0, "last_updated": time.strftime("%Y-%m-%d %H:%M:%S")}}

def save_feedback_database(data):
    """Save feedback data to JSON file database."""
    try:
        with open(FEEDBACK_DB_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Error saving database: {e}")
        return False

def add_feedback_to_database(rating, comment):
    """Add new feedback to the database and update metadata."""
    database = load_feedback_database()

    new_entry = {
        "id": len(database["feedback_entries"]) + 1,
        "rating": rating,
        "comment": comment.strip(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    database["feedback_entries"].append(new_entry)

    # Update metadata
    total_entries = len(database["feedback_entries"])
    total_rating = sum(entry["rating"] for entry in database["feedback_entries"])
    average_rating = total_rating / total_entries if total_entries > 0 else 0

    database["metadata"] = {
        "total_feedback": total_entries,
        "average_rating": round(average_rating, 2),
        "last_updated": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    if save_feedback_database(database):
        return True, database
    else:
        return False, database

# --- TRIVIA QUIZ STATE ---
TRIVIA_SESSIONS = {}  # Store quiz sessions by session_id

# --- FIX FOR COLAB SECRETS LOADING ---
try:
    # Import the Colab utility to access the stored secrets
    from google.colab import userdata

    # Securely fetch the secret value and explicitly set the environment variable.
    # This ensures os.environ.get() works correctly in Colab.
    secret_value = userdata.get("GROQ_API_KEY")
    if secret_value:
        os.environ["GROQ_API_KEY"] = secret_value
except ImportError:
    pass
# -----------------------------------

# --- Configuration ---
GROQ_CHAT_MODEL = "llama-3.1-8b-instant"
GROQ_WHISPER_MODEL = "whisper-large-v3"
API_KEY = os.environ.get("GROQ_API_KEY")

# --- System Prompts ---
DEFAULT_LANGUAGE= "English"
TOURISM_EXPERT_SYSTEM_PROMPT = (
    "You are 'WanderBot', a world-class, friendly, and enthusiastic tourism expert. "
    "Your goal is to provide detailed, helpful, and inspiring information about travel destinations. "
    f"Your DEFAULT language is {DEFAULT_LANGUAGE}. "
    "ONLY switch languages if the user clearly writes in that language. "
    f"If the user writes in {DEFAULT_LANGUAGE}, ALWAYS reply in {DEFAULT_LANGUAGE}. "
    "If another language is detected with high certainty, reply fully in that language. "
    "If a question is not about tourism, politely redirect the user back to travel topics."
)


CULTURE_SYSTEM_PROMPT = (
    f"You are a specialized Cultural and Historical Analyst, tasked with generating a detailed, "
    f"objective, and encyclopedic report on a specific topic. Your response must be highly factual "
    f" and structured like a Wikipedia entry. Use the {GROQ_CHAT_MODEL} model's vast knowledge base to "
    f"synthesize this information."
    f"\n\n**INSTRUCTIONS:**"
    f"\n1. **Structure:** The output must be formatted with a main title (markdown #), a brief introduction, and 3-4 structured paragraphs. Use bold text for key terms."
    f"\n2. **Tone:** Maintain a purely academic, informative, and neutral tone. Do not use any conversational language (like greetings or sign-offs)."
)

ROUTE_SYSTEM_PROMPT = (
    "You are an expert transport and logistics analyst. Given a starting point and destination, "
    "provide a realistic, estimated travel time and distance for a typical driving route. "
    "Do not include any map links or conversational filler. State the distance first, then the time, in a single, concise paragraph."
)

ITINERARY_SYSTEM_PROMPT = """
You are an expert itinerary planner. Your task is to generate a comprehensive travel plan strictly in JSON format.
DO NOT INCLUDE ANY TEXT, MARKDOWN OUTSIDE OF THE JSON BLOCK.

The JSON object must adhere to the following schema:
{
  "destination": "string",
  "total_days": "integer",
  "trip_focus": "string (e.g., Hiking, History, Relaxation)",
  "daily_plan": [
    {
      "day": "integer",
      "theme": "string (e.g., Ancient History Exploration, Mountain Views)",
      "morning": "string (e.g., Visit the Acropolis)",
      "afternoon": "string (e.g., Lunch and visit the Plaka district)",
      "evening": "string (e.g., Traditional Greek dinner with live music)"
    }
    // Include an object for each day up to total_days
  ]
}

Ensure the final output is ONLY the raw JSON string, enclosed in a single JSON block.
"""

# NEW PROMPT FOR BUDGET ESTIMATOR
BUDGET_SYSTEM_PROMPT = """
You are a professional travel cost analyst. Your task is to estimate the daily and total budget for a trip based on the provided destination and travel style.
DO NOT INCLUDE ANY TEXT OR MARKDOWN OUTSIDE OF THE JSON BLOCK.

The JSON object must adhere strictly to the following schema, with all currency values in USD ($):
{
  "destination": "string",
  "travel_style": "string (e.g., Budget, Mid-Range, Luxury)",
  "estimated_daily_budget": {
    "accommodation": "number (USD estimate)",
    "food_and_dining": "number (USD estimate)",
    "activities_and_fees": "number (USD estimate)",
    "local_transport": "number (USD estimate)",
    "miscellaneous": "number (USD estimate)"
  },
  "notes": "string (A brief 1-2 sentence note on why the cost is high or low)"
}

Ensure the final output is ONLY the raw JSON string, enclosed in a single JSON block.
"""

# TRIVIA QUIZ SYSTEM PROMPT
TRIVIA_SYSTEM_PROMPT = """
You are a travel trivia expert. Generate multiple-choice questions about world travel, destinations, cultures, landmarks, and geography.

Generate exactly ONE multiple-choice question in JSON format with the following structure:
{
  "question": "string",
  "options": ["A. option1", "B. option2", "C. option3", "D. option4"],
  "correct_answer": "A"  // or B, C, D
}

The question should be about travel, tourism, world geography, famous landmarks, cultural facts, or travel history.
Make it engaging and educational. The difficulty should be moderate.

Return ONLY the JSON, no additional text.
"""

# --- Language & Currency Lists ---
LANGUAGES = [
    "English", "Spanish", "French", "German", "Italian", "Portuguese",
    "Japanese", "Korean", "Mandarin Chinese", "Hindi", "Russian", "Arabic"
]

SIMULATED_RATES = {
    "USD": 1.0, "EUR": 0.92, "GBP": 0.79, "JPY": 156.9, "CAD": 1.37, "AUD": 1.51, "INR": 83.3, "AED": 3.67, "SAR": 3.75, "KRW": 1374.5
}
CURRENCY_CODES = list(SIMULATED_RATES.keys())


# --- Groq Client Initialization ---
client = None
if API_KEY:
    try:
        client = Groq(api_key=API_KEY)
    except Exception as e:
        print(f"Error initializing Groq client: {e}")
        client = None

# ----------------------------------------------------------------------
# 1. Tourism Chatbot Function
# ----------------------------------------------------------------------

def groq_chat(message, history):
    """
    Handles the standard multilingual conversational exchange for the Tourism Chatbot.
    """
    if client is None:
        yield "Error: The Groq API client is not initialized."
        return

    messages = [{"role": "system", "content": TOURISM_EXPERT_SYSTEM_PROMPT}]

    for human, assistant in history:
        messages.append({"role": "user", "content": human})
        messages.append({"role": "assistant", "content": assistant})

    messages.append({"role": "user", "content": message})

    try:
        chat_completion = client.chat.completions.create(
            messages=messages,
            model=GROQ_CHAT_MODEL,
            stream=True
        )

        response_content = ""
        for chunk in chat_completion:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                response_content += chunk.choices[0].delta.content
                yield response_content

    except APIError as e:
        yield f"**[API Error]** Groq Chatbot failed: {e}. Check API key and rate limits."
    except Exception as e:
        yield f"**[Error]** An unexpected error occurred: {e}"

# ----------------------------------------------------------------------
# 2. Audio Translator Functions
# ----------------------------------------------------------------------

def groq_transcribe(audio_filepath):
    """
    Converts audio input to text using the Groq Whisper model.
    """
    if not audio_filepath:
        return None, "Error: Please upload or record audio first."
    if client is None:
        return None, "Error: Groq client not initialized."

    try:
        with open(audio_filepath, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model=GROQ_WHISPER_MODEL,
                file=audio_file
            )
        return transcript.text, None
    except APIError as e:
        return None, f"**[Transcription API Error]** Failed to transcribe audio: {e}"
    except Exception as e:
        return None, f"**[Error]** An unexpected error occurred during transcription: {e}"

def groq_translate_text(text, source_lang, target_lang):
    """
    Translates text using the Groq LLM.
    """
    if not text:
        return None, "Error: Transcription failed, no text to translate."
    if client is None:
        return None, "Error: Groq client not initialized."

    translation_prompt = (
        f"You are a professional, highly accurate language translator. "
        f"Translate the following text from **{source_lang}** to **{target_lang}**. "
        f"Only return the translated text, with no extra commentary or formatting."
    )

    messages = [
        {"role": "system", "content": translation_prompt},
        {"role": "user", "content": text}
    ]

    try:
        chat_completion = client.chat.completions.create(
            messages=messages,
            model=GROQ_CHAT_MODEL
        )
        return chat_completion.choices[0].message.content.strip(), None
    except APIError as e:
        return None, f"**[Translation API Error]** Failed to translate text: {e}"
    except Exception as e:
        return None, f"**[Error]** An unexpected error occurred during translation: {e}"


def translate_pipeline(audio_filepath, source_lang, target_lang):
    """
    Full pipeline: S2T -> T2T.
    """
    transcribed_text, error = groq_transcribe(audio_filepath)
    if error:
        return "", error, ""

    translated_text, error = groq_translate_text(transcribed_text, source_lang, target_lang)
    if error:
        return transcribed_text, error, ""

    return transcribed_text, "", translated_text


# ----------------------------------------------------------------------
# 3. Culture & Tradition Function
# ----------------------------------------------------------------------

def fetch_culture_info(place, topic):
    """
    Generates a factual report by prompting the LLM to act as a historical analyst.
    """
    if client is None:
        return f"Error: Groq client not initialized."

    user_query = f"Generate a comprehensive report on the {topic} of {place}. Start with a title and structured content."

    messages = [
        {"role": "system", "content": CULTURE_SYSTEM_PROMPT.format(topic=topic, place=place)},
        {"role": "user", "content": user_query}
    ]

    try:
        chat_completion = client.chat.completions.create(
            messages=messages,
            model=GROQ_CHAT_MODEL
        )
        return chat_completion.choices[0].message.content
    except APIError as e:
        return f"**[API Error]** Factual retrieval failed: {e}. Check API key and rate limits."
    except Exception as e:
        return f"**[Error]** An unexpected error occurred: {e}"

# ----------------------------------------------------------------------
# 4. Itinerary Planner Function
# ----------------------------------------------------------------------

def generate_itinerary(destination, total_days, trip_focus):
    """
    Generates a structured JSON itinerary and converts it to Markdown.
    """
    if client is None:
        return f"Error: Groq client not initialized."

    if not destination or not total_days:
        return f"Error: Please provide a destination and number of days."

    user_query = (
        f"Create a travel itinerary for: "
        f"Destination: {destination}, "
        f"Total Days: {total_days}, "
        f"Focus: {trip_focus}. "
        f"Adhere strictly to the JSON schema provided in the system prompt."
    )

    messages = [
        {"role": "system", "content": ITINERARY_SYSTEM_PROMPT},
        {"role": "user", "content": user_query}
    ]

    try:
        chat_completion = client.chat.completions.create(
            messages=messages,
            model=GROQ_CHAT_MODEL,
        )
        raw_json_string = chat_completion.choices[0].message.content.strip()

        # Clean up Markdown/JSON wrappers
        if raw_json_string.startswith("```json"):
            raw_json_string = raw_json_string.strip("```json").strip("```").strip()

        itinerary_data = json.loads(raw_json_string)

        # Format JSON output as human-readable Markdown
        markdown_output = f"# ✈️ {itinerary_data.get('destination', 'Trip')} Itinerary ({itinerary_data.get('total_days', '')} Days)\n"
        markdown_output += f"**Focus:** {itinerary_data.get('trip_focus', 'General')}\n\n"

        for day_plan in itinerary_data.get('daily_plan', []):
            markdown_output += f"## Day {day_plan.get('day', '?')}: {day_plan.get('theme', 'Activities')}\n"
            markdown_output += f"- **Morning:** {day_plan.get('morning', 'N/A')}\n"
            markdown_output += f"- **Afternoon:** {day_plan.get('afternoon', 'N/A')}\n"
            markdown_output += f"- **Evening:** {day_plan.get('evening', 'N/A')}\n\n"

        return markdown_output

    except json.JSONDecodeError:
        return f"**[Error]** Could not parse JSON response from the model. Raw output:\n\n```json\n{raw_json_string}\n```"
    except APIError as e:
        return f"**[API Error]** Itinerary generation failed: {e}. Check API key and rate limits."
    except Exception as e:
        return f"**[Error]** An unexpected error occurred: {e}"

# ----------------------------------------------------------------------
# 5. Currency Converter Function (Stable, Direct Logic)
# ----------------------------------------------------------------------

def perform_conversion(amount, from_currency, to_currency):
    """
    Performs direct arithmetic conversion using hardcoded simulated rates.
    This bypasses the LLM for stability.
    """
    try:
        # Robustly convert input to float
        amount = float(amount)
        if amount <= 0:
            return "Error: Amount must be a positive number."
    except ValueError:
        return "Error: Invalid amount entered. Please enter a valid number."

    if from_currency not in SIMULATED_RATES or to_currency not in SIMULATED_RATES:
        return "Error: Selected currency code is not supported."

    rate_from = SIMULATED_RATES[from_currency]
    rate_to = SIMULATED_RATES[to_currency]

    # Convert to base USD, then convert to target currency
    base_amount = amount / rate_from
    converted_amount = base_amount * rate_to

    # Format the output nicely
    return (
        f"Conversion Result:\n\n"
        f"**Original Amount:** {amount:,.2f} {from_currency}\n"
        f"**Exchange Rate (1 {from_currency} = {rate_to / rate_from:.4f} {to_currency})**\n"
        f"**Converted Amount:** {converted_amount:,.2f} {to_currency}\n\n"
        f"*Note: Rates are simulated for demonstration purposes.*"
    )

# ----------------------------------------------------------------------
# 6. Route Planner Function
# ----------------------------------------------------------------------

def generate_route_and_map(origin, destination, mode="driving"):
    """
    Generates a factual report by prompting the LLM to act as a historical analyst.
    """
    if client is None:
        return "Error: Groq client not initialized."

    if not origin or not destination:
        return "Error: Please provide both an origin and a destination."

    # 1. Get Estimated Travel Time/Distance from LLM (Simulated RAG)
    user_query = f"Estimate the travel time and distance for a typical route from {origin} to {destination} using {mode} mode."

    messages = [
        {"role": "system", "content": ROUTE_SYSTEM_PROMPT},
        {"role": "user", "content": user_query}
    ]

    try:
        chat_completion = client.chat.completions.create(
            messages=messages,
            model=GROQ_CHAT_MODEL
        )
        travel_estimate = chat_completion.choices[0].message.content.strip()
    except APIError as e:
        travel_estimate = f"**[API Error]** Could not get travel estimate: {e}"
    except Exception as e:
        travel_estimate = f"**[Error]** An unexpected error occurred: {e}"

    # 2. Generate Google Maps URLs
    encoded_origin = quote_plus(origin)
    encoded_destination = quote_plus(destination)

    # Direct Google Maps URL for directions
    direct_map_url = (
        f"https://www.google.com/maps/dir/?api=1&origin={encoded_origin}&"
        f"destination={encoded_destination}&"
        f"travelmode={mode}"
    )
# Google Maps Embed URL for iframe
    embed_map_url = (
        f"https://www.google.com/maps?q={encoded_origin}"
        f"+to+{encoded_destination}"
        f"&output=embed"
    )

    iframe_html = f'''<iframe width="100%" height="450" style="border:0; border-radius: 8px;"
            loading="lazy"
            allowfullscreen
            referrerpolicy="no-referrer-when-downgrade"
            src="{embed_map_url}">
    </iframe>
    '''

    # 3. Combined Markdown Output
    markdown_output = (
        f"# 🗺 Route from {origin} to {destination}\n\n"
        f"**Estimated Travel Details (LLM Simulation):** {travel_estimate}\n\n"
        f"## Live Interactive Map\n"
        f"To view the route and turn-by-turn directions, you can click [here]({direct_map_url}) or use the embedded map below.\n"
        f"{iframe_html}"
    )

    return markdown_output

# ----------------------------------------------------------------------
# 7. Budget Estimator Function (NEW)
# ----------------------------------------------------------------------

def generate_budget(destination, travel_style):
    """
    Generates a structured JSON budget estimate and converts it to a table.
    """
    if client is None:
        return f"Error: Groq client not initialized."

    if not destination or not travel_style:
        return f"Error: Please provide a destination and a travel style."

    user_query = (
        f"Estimate the daily budget for: "
        f"Destination: {destination}, "
        f"Travel Style: {travel_style}. "
        f"Adhere strictly to the JSON schema provided in the system prompt."
    )

    messages = [
        {"role": "system", "content": BUDGET_SYSTEM_PROMPT},
        {"role": "user", "content": user_query}
    ]

    try:
        chat_completion = client.chat.completions.create(
            messages=messages,
            model=GROQ_CHAT_MODEL,
        )
        raw_json_string = chat_completion.choices[0].message.content.strip()

        # Clean up Markdown/JSON wrappers
        if raw_json_string.startswith("```json"):
            raw_json_string = raw_json_string.strip("```json").strip("```").strip()

        budget_data = json.loads(raw_json_string)

        daily_budget = budget_data.get('estimated_daily_budget', {})
        total_daily_cost = sum(daily_budget.values())

        # Format JSON output as human-readable Markdown table
        markdown_output = f"# 💰 Daily Budget Estimate for {budget_data.get('destination', 'Destination')} ({budget_data.get('travel_style', 'Style')})\n\n"

        markdown_output += f"| Category | Daily Cost (USD) |\n"
        markdown_output += f"| :--- | :---: |\n"
        for category, cost in daily_budget.items():
            markdown_output += f"| {category.replace('_', ' ').title()} | ${cost:,.2f} |\n"

        markdown_output += f"| **TOTAL ESTIMATED DAILY COST** | **${total_daily_cost:,.2f}** |\n\n"
        markdown_output += f"**Analyst Notes:** {budget_data.get('notes', 'No specific notes provided.')}\n"

        return markdown_output

    except json.JSONDecodeError:
        return f"**[Error]** Could not parse JSON response from the model. Raw output:\n\n```json\n{raw_json_string}\n```"
    except APIError as e:
        return f"**[API Error]** Budget estimation failed: {e}. Check API key and rate limits."
    except Exception as e:
        return f"**[Error]** An unexpected error occurred: {e}"

# ----------------------------------------------------------------------
# 8. Travel Trivia Quiz Functions (NEW)
# ----------------------------------------------------------------------

def generate_trivia_question(destination=None):
    """
    Generates a single travel trivia question using Groq API.
    """
    if client is None:
        return None, "Error: Groq client not initialized."

    prompt = TRIVIA_SYSTEM_PROMPT
    if destination:
        prompt += f"\n\nFocus the question on {destination} or related travel topics."

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": "Generate one travel trivia question."}
    ]

    try:
        chat_completion = client.chat.completions.create(
            messages=messages,
            model=GROQ_CHAT_MODEL,
        )
        raw_json_string = chat_completion.choices[0].message.content.strip()

        # Clean up JSON response
        if raw_json_string.startswith("```json"):
            raw_json_string = raw_json_string.strip("```json").strip("```").strip()

        question_data = json.loads(raw_json_string)
        return question_data, None

    except json.JSONDecodeError:
        return None, f"Could not parse question. Raw response: {raw_json_string}"
    except APIError as e:
        return None, f"API Error: {e}"
    except Exception as e:
        return None, f"Error: {e}"

def start_trivia_quiz(destination):
    """
    Starts a new trivia quiz session.
    """
    session_id = str(time.time())  # Simple session ID

    # Generate first question
    question_data, error = generate_trivia_question(destination)

    if error:
        return None, None, None, None, f"Error starting quiz: {error}"

    # Initialize session
    TRIVIA_SESSIONS[session_id] = {
        'destination': destination,
        'current_question': question_data,
        'score': 0,
        'questions_asked': 1,
        'total_questions': 5,  # Fixed number of questions per quiz
        'start_time': time.time()
    }

    # Format question display
    question_display = f"## 🧠 Question 1/{TRIVIA_SESSIONS[session_id]['total_questions']}\n\n"
    question_display += f"**{question_data['question']}**\n\n"

    for option in question_data['options']:
        question_display += f"- {option}\n"

    return session_id, question_display, gr.update(visible=True), gr.update(visible=False), ""

def submit_trivia_answer(session_id, user_answer, current_display):
    """
    Processes user's answer and provides feedback, then loads next question.
    """
    if session_id not in TRIVIA_SESSIONS:
        return None, None, None, "Quiz session expired. Please start a new quiz."

    session = TRIVIA_SESSIONS[session_id]
    current_question = session['current_question']

    # Check answer
    user_answer_clean = user_answer.upper().strip()
    correct_answer = current_question['correct_answer']

    # Prepare feedback
    feedback = ""
    if user_answer_clean == correct_answer:
        session['score'] += 1
        feedback = f"✅ **Correct!** Well done!\n\n"
    else:
        feedback = f"❌ **Incorrect!** The right answer was **{correct_answer}**\n\n"

    feedback += f"**Explanation:** {current_question.get('explanation', 'No explanation provided.')}\n\n"

    # Check if quiz is complete
    if session['questions_asked'] >= session['total_questions']:
        # Quiz complete - show final results
        final_score = session['score']
        total_questions = session['total_questions']
        percentage = (final_score / total_questions) * 100

        final_display = f"# 🎉 Quiz Complete!\n\n"
        final_display += f"**Final Score: {final_score}/{total_questions} ({percentage:.1f}%)**\n\n"

        if percentage >= 80:
            final_display += "🏆 **Excellent!** You're a travel expert!\n"
        elif percentage >= 60:
            final_display += "👍 **Good job!** You know your travel facts!\n"
        else:
            final_display += "📚 **Keep exploring!** The world is full of amazing facts to discover!\n"

        final_display += f"\n*Quiz about: {session['destination'] if session['destination'] else 'General Travel'}*"

        # Clean up session
        del TRIVIA_SESSIONS[session_id]

        return None, final_display, gr.update(visible=False), gr.update(visible=True), ""

    else:
        # Generate next question
        next_question_data, error = generate_trivia_question(session['destination'])

        if error:
            return session_id, current_display + f"\n\nError loading next question: {error}", gr.update(visible=True), gr.update(visible=False), feedback

        session['current_question'] = next_question_data
        session['questions_asked'] += 1

        # Format next question
        next_display = f"## 🧠 Question {session['questions_asked']}/{session['total_questions']}\n\n"
        next_display += f"**{next_question_data['question']}**\n\n"

        for option in next_question_data['options']:
            next_display += f"- {option}\n"

        next_display += f"\n---\n{feedback}---\n"

        return session_id, next_display, gr.update(visible=True), gr.update(visible=False), ""

# ----------------------------------------------------------------------
# 9. Enhanced Public Feedback System with Star Ratings & Database
# ----------------------------------------------------------------------

def load_feedback():
    """Reads and formats all stored feedback for display with statistics."""
    database = load_feedback_database()
    feedback_entries = database.get("feedback_entries", [])
    metadata = database.get("metadata", {})

    if not feedback_entries:
        return "**No feedback yet! Be the first to share your experience.**", "⭐ Overall Rating: No ratings yet"

    # Format statistics
    total_feedback = metadata.get("total_feedback", 0)
    average_rating = metadata.get("average_rating", 0)

    stats_display = f"⭐ **Overall Rating: {average_rating}/5** ({total_feedback} reviews)"

    # Create star visualization for average rating
    stars_full = "★" * int(average_rating)
    stars_empty = "☆" * (5 - int(average_rating))
    decimal_part = average_rating - int(average_rating)

    if decimal_part >= 0.75:
        stars_visual = stars_full + "★" + stars_empty[1:]
    elif decimal_part >= 0.25:
        stars_visual = stars_full + "½" + stars_empty[1:]
    else:
        stars_visual = stars_full + stars_empty

    stats_display += f"\n{stars_visual}\n"

    # Format feedback entries
    markdown_output = f"## 🌟 User Reviews & Ratings\n\n"
    markdown_output += f"{stats_display}\n\n"
    markdown_output += "---\n\n"

    # Display newest feedback first
    for fb in reversed(feedback_entries):
        stars = "★" * fb['rating'] + "☆" * (5 - fb['rating'])
        markdown_output += (
            f"**{stars}** ({fb['rating']}/5)\n\n"
            f"**💬 Comment:** {fb['comment']}\n\n"
            f"*(Submitted: {fb['timestamp']})*\n"
            f"---\n"
        )

    return markdown_output, stats_display

def star_rating_component():
    """Creates a star rating component using radio buttons."""
    return gr.Radio(
        choices=["1", "2", "3", "4", "5"],
        label="⭐ Rate Your Experience (1-5 Stars)",
        info="Select number of stars: 1=Poor, 5=Excellent",
        value="5"
    )

def add_feedback(rating, comment):
    """
    Writes new feedback to the database and updates the UI.
    Returns: (message_box_update, rating_input_update, comment_input_update,
              feedback_display_update, stats_display_update)
    """
    if not comment.strip():
        # message_box_update, rating_input_update, comment_input_update, feedback_display_update, stats_display_update
        return (
            gr.update(value="⚠️ **Please leave a comment before submitting feedback.** ⚠️", visible=True),
            gr.update(value="5"),
            gr.update(value="", placeholder="What did you like or what can be improved?"),
            gr.no_change(), # feedback_display_component does not change on validation error
            gr.no_change()  # stats_display_component does not change on validation error
        )

    # Convert rating to integer
    try:
        rating_int = int(rating)
    except ValueError:
        return (
            gr.update(value="⚠️ **Invalid rating selected.** ⚠️", visible=True),
            gr.update(value="5"),
            gr.update(value="", placeholder="What did you like or what can be improved?"),
            gr.no_change(),
            gr.no_change()
        )

    # Add to database
    success, updated_database = add_feedback_to_database(rating_int, comment)

    if not success:
        return (
            gr.update(value="⚠️ **Error saving feedback. Please try again.** ⚠️", visible=True),
            gr.update(value="5"),
            gr.update(value="", placeholder="What did you like or what can be improved?"),
            gr.no_change(),
            gr.no_change()
        )

    # Returns updated feedback list and a success message
    thank_you_message = "✅ **Thank you! Your feedback has been successfully submitted and is now public!** ✅"

    # Load updated feedback for display components
    updated_feedback_display_str, updated_stats_display_str = load_feedback()

    # Return updates for all five output components
    return (
        gr.update(value=thank_you_message, visible=True), # message_box
        gr.update(value="5"), # rating_input
        gr.update(value="", placeholder="What did you like or what can be improved?"), # comment_input
        gr.update(value=updated_feedback_display_str), # feedback_display_component
        gr.update(value=updated_stats_display_str) # stats_display_component
    )


# ----------------------------------------------------------------------
# 10. Gradio Interface Setup (UPDATED WITH ENHANCED FEEDBACK SYSTEM)
# ----------------------------------------------------------------------

# Custom CSS for light contrasting colors and title styling
custom_css = """
:root {
    --primary-color: #2563eb;      /* Bright blue */
    --secondary-color: #7c3aed;    /* Purple */
    --accent-color: #dc2626;       /* Red */
    --success-color: #059669;      /* Green */
    --warning-color: #d97706;      /* Amber */
    --light-bg: #f8fafc;           /* Very light gray */
    --card-bg: #ffffff;            /* White */
    --text-dark: #1e293b;          /* Dark blue-gray */
    --text-light: #64748b;         /* Medium gray */
    --border-color: #e2e8f0;       /* Light gray border */
}

.gradio-container {
    background: linear-gradient(135deg, var(--light-bg) 0%, #e0f2fe 100%) !important;
    font-family: 'Segoe UI', system-ui, sans-serif !important;
}

.main-title {
    text-align: center !important;
    font-size: 3.5rem !important;
    font-weight: 900 !important;
    background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    margin-bottom: 1rem !important;
    padding: 1rem !important;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.1) !important;
}

.subtitle {
    text-align: center !important;
    font-size: 1.2rem !important;
    color: var(--text-light) !important;
    margin-bottom: 2rem !important;
    font-weight: 400 !important;
}

.tab-nav {
    background: var(--card-bg) !important;
    border-radius: 12px !important;
    padding: 8px !important;
    margin: 10px !important;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important;
}

.tab-nav button {
    border-radius: 8px !important;
    margin: 2px !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
}

.tab-nav button.selected {
    background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%) !important;
    color: white !important;
}

.contain {
    background: var(--card-bg) !important;
    border-radius: 16px !important;
    border: 1px solid var(--border-color) !important;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1) !important;
    margin: 10px !important;
}

.button-primary {
    background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    transition: all 0.3s ease !important;
}

.button-primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 20px -4px rgba(37, 99, 235, 0.4) !important;
}

.button-secondary {
    background: var(--light-bg) !important;
    border: 2px solid var(--primary-color) !important;
    color: var(--primary-color) !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
}

.prose h1, .prose h2, .prose h3 {
    color: var(--text-dark) !important;
    font-weight: 700 !important;
}

.markdown-text {
    color: var(--text-dark) !important;
}

.gr-box {
    border: 2px solid var(--border-color) !important;
    border-radius: 12px !important;
    background: var(--light-bg) !important;
}

.gr-input, .gr-textbox, .gr-dropdown {
    border-radius: 10px !important;
    border: 2px solid var(--border-color) !important;
    background: var(--card-bg) !important;
}

.gr-input:focus, .gr-textbox:focus, .gr-dropdown:focus {
    border-color: var(--primary-color) !important;
    box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1) !important;
}

.success-message {
    background: linear-gradient(135deg, var(--success-color) 0%, #10b981 100%) !important;
    color: white !important;
    padding: 1rem !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
}

.warning-message {
    background: linear-gradient(135deg, var(--warning-color) 0%, #f59e0b 100%) !important;
    color: white !important;
    padding: 1rem !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
}

.error-message {
    background: linear-gradient(135deg, var(--accent-color) 0%, #ef4444 100%) !important;
    color: white !important;
    padding: 1rem !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
}
"""

if client is None:
    # Display error if the API key wasn't loaded
    interface_content = gr.Markdown(
        """
        # ⚠️ Groq API Key Error ⚠️

        **The application could not start because the `GROQ_API_KEY` was not found.**

        Please ensure the following in your Google Colab environment:
        1. You have installed the required libraries: `!pip install groq gradio`
        2. You have saved your Groq API key in the **Secrets** panel on the left sidebar.
        3. The secret is named **`GROQ_API_KEY`**.
        4. **Notebook access** is enabled for the secret.
        """
    )
else:
    # --- Interface Definitions ---

    # 1. Chatbot Tab
    chatbot_interface = gr.ChatInterface(
        fn=groq_chat,
        title="",
        description=f"Ask WanderBot anything about travel! (Model: {GROQ_CHAT_MODEL})",
        submit_btn="Ask WanderBot"
    )

    # 2. Translator Tab
    translator_interface = gr.Interface(
        fn=translate_pipeline,
        title="",
        live=False,
        submit_btn="Translate Audio",
        description=f"Convert speech to text, then translate it between two languages. Uses Groq Whisper ({GROQ_WHISPER_MODEL}) and Llama 3.1 ({GROQ_CHAT_MODEL}).",
        inputs=[
            gr.Audio(type="filepath", format="wav", label="1. Speak or Upload Audio (Max 25MB)", sources=["microphone", "upload"]),
            gr.Dropdown(label="2. Source Language", choices=LANGUAGES, value="English"),
            gr.Dropdown(label="3. Target Language", choices=LANGUAGES, value="Spanish"),
        ],
        outputs=[
            gr.Textbox(label="4. Transcribed Text (Speech to Text)", lines=3),
            gr.Textbox(label="Error Messages (if any)", lines=1),
            gr.Textbox(label="5. Translated Text (Text to Text)", lines=5)
        ]
    )

    # 3. Culture Tab
    culture_interface = gr.Interface(
        fn=fetch_culture_info,
        title="",
        live=False,
        submit_btn="Generate Report",
        description=f"Instantly retrieve a detailed, structured report on the History, Culture, or Tradition of any place. Uses Llama 3.1 ({GROQ_CHAT_MODEL}).",
        inputs=[
            gr.Textbox(label="1. Enter City, Country, or Region (e.g., Kyoto, Japan)", lines=1, placeholder="e.g., The Amazon Rainforest"),
            gr.Radio(label="2. Select Topic", choices=["Tradition", "Culture", "History"], value="Culture"),
        ],
        outputs=[
            gr.Markdown(label="3. Factual Report (Simulated Wikipedia Lookup)")
        ]
    )

    # 4. Itinerary Planner Tab
    itinerary_interface = gr.Interface(
        fn=generate_itinerary,
        title="",
        live=False,
        submit_btn="Generate Itinerary",
        description=f"Create a structured, day-by-day travel plan. The LLM is forced to output JSON for clean results.",
        inputs=[
            gr.Textbox(label="1. Destination", lines=1, placeholder="e.g., Rome, Italy"),
            gr.Slider(label="2. Number of Days", minimum=1, maximum=10, step=1, value=3),
            gr.Textbox(label="3. Trip Focus (e.g., Food, Hiking, Museums)", lines=1, placeholder="e.g., Food and Ancient History"),
        ],
        outputs=[
            gr.Markdown(label="4. Structured Itinerary (Generated via JSON)")
        ]
    )

    # 5. Budget Estimator Tab (NEW)
    budget_interface = gr.Interface(
        fn=generate_budget,
        title="",
        live=False,
        submit_btn="Estimate Budget",
        description=f"Generate a structured daily budget estimate based on destination and travel style.",
        inputs=[
            gr.Textbox(label="1. Destination (City/Country)", lines=1, placeholder="e.g., London, UK"),
            gr.Dropdown(label="2. Travel Style", choices=["Budget", "Mid-Range", "Luxury"], value="Mid-Range"),
        ],
        outputs=[
            gr.Markdown(label="3. Estimated Daily Costs (USD)")
        ]
    )

    # 6. Currency Converter Tab
    currency_converter_interface = gr.Interface(
        fn=perform_conversion,
        title="",
        live=False,
        submit_btn="Convert Amount",
        description="Convert currency instantly using internal (simulated) exchange rates. No Groq API calls needed for this calculation.",
        inputs=[
            gr.Number(label="1. Amount to Convert", value=100),
            gr.Dropdown(label="2. From Currency (Source)", choices=CURRENCY_CODES, value="USD"),
            gr.Dropdown(label="3. To Currency (Target)", choices=CURRENCY_CODES, value="EUR"),
        ],
        outputs=[
            gr.Markdown(label="4. Conversion Result")
        ]
    )

    # 7. Route Planner Tab
    route_interface = gr.Interface(
        fn=generate_route_and_map,
        title="",
        live=False,
        
        submit_btn="Find Route",
        description="Get estimated travel details (LLM) and a live map route (Google Maps link) between two places.",
        inputs=[
            gr.Textbox(label="1. Starting Point", lines=1, placeholder="e.g., Eiffel Tower, Paris"),
            gr.Textbox(label="2. Destination", lines=1, placeholder="e.g., Louvre Museum, Paris"),
            gr.Radio(label="3. Travel Mode", choices=["driving", "walking", "transit", "bicycling"], value="driving")
        ],
        outputs=[
            gr.Markdown(label="4. Estimated Route Details & Interactive Map",sanitize_html=False)
        ]
    )

    # 8. Travel Trivia Quiz Tab (NEW)
    with gr.Blocks() as trivia_quiz_interface:
        gr.Markdown("# 🧠 Travel Trivia Quiz")
        gr.Markdown("Test your travel knowledge with our interactive quiz! Answer 5 questions and see how much you know about world travel.")

        with gr.Row():
            with gr.Column(scale=2):
                destination_input = gr.Textbox(
                    label="Optional: Focus on specific destination",
                    placeholder="e.g., Japan, Paris, or leave blank for general travel questions",
                    lines=1
                )
                start_btn = gr.Button("🎯 Start New Quiz", variant="primary")

            with gr.Column(scale=3):
                quiz_output = gr.Markdown(
                    "### Ready to test your travel knowledge?\n\nClick 'Start New Quiz' to begin!",
                    label="Quiz Display"
                )

        with gr.Row(visible=False) as answer_section:
            with gr.Column():
                answer_input = gr.Textbox(
                    label="Your Answer (Enter A, B, C, or D)",
                    placeholder="Enter A, B, C, or D",
                    max_lines=1
                )
                submit_btn = gr.Button("📝 Submit Answer", variant="secondary")

        feedback_output = gr.Markdown("", label="Feedback")

        # Session state (hidden)
        session_state = gr.State()

        # Event handlers
        start_btn.click(
            fn=start_trivia_quiz,
            inputs=[destination_input],
            outputs=[session_state, quiz_output, answer_section, start_btn, feedback_output]
        )

        submit_btn.click(
            fn=submit_trivia_answer,
            inputs=[session_state, answer_input, quiz_output],
            outputs=[session_state, quiz_output, answer_section, start_btn, feedback_output]
        ).then(
            lambda: "",  # Clear answer input after submission
            outputs=[answer_input]
        )

    # 9. Enhanced Public Feedback System with Star Ratings
    feedback_display_component = gr.Markdown(label="Community Reviews")
    stats_display_component = gr.Markdown(label="Current Statistics")

    with gr.Blocks() as feedback_blocks:
        gr.Markdown("# ⭐ Public Feedback Center")
        gr.Markdown("Share your experience and see what others think about our travel assistant!")

        with gr.Tab("✍️ Give Feedback"):
            gr.Markdown("## ✨ Share Your Experience")
            gr.Markdown("Your feedback helps us improve WanderBot! All reviews are public and visible to everyone.")

            with gr.Row():
                with gr.Column(scale=1):
                    rating_input = star_rating_component()

                with gr.Column(scale=2):
                    comment_input = gr.Textbox(
                        label="💬 Your Comments",
                        lines=3,
                        placeholder="What did you like about WanderBot? What can be improved? Share your experience...",
                        max_lines=5
                    )

            message_box = gr.Markdown("Status: Ready to submit your feedback.", visible=True)
            submit_button = gr.Button("⭐ Submit Feedback", variant="primary")

            stats_display_component.render()

            submit_button.click(
                fn=add_feedback,
                inputs=[rating_input, comment_input],
                outputs=[message_box, rating_input, comment_input, feedback_display_component, stats_display_component],
            )

        with gr.Tab("📊 See All Reviews"):
            gr.Markdown("## 🌟 Community Reviews")
            gr.Markdown("See what other travelers think about WanderBot!")

            feedback_display_component.render()

        feedback_blocks.load(
            lambda: load_feedback(),
            outputs=[feedback_display_component, stats_display_component]
        )

    # --- Main Tabbed Interface with Custom Styling ---
    with gr.Blocks( title="Zenix Travel Companion") as interface_content:
        gr.Markdown("# <div class='main-title'>Zenix Travel Companion</div>")
        gr.Markdown("<div class='subtitle'>Your AI-Powered Guide to World Exploration</div>")

        gr.TabbedInterface(
            [
                chatbot_interface,
                translator_interface,
                culture_interface,
                itinerary_interface,
                budget_interface,
                currency_converter_interface,
                route_interface,
                trivia_quiz_interface,
                feedback_blocks
            ],
            [
                "🌍 Tourism Chatbot",
                "🗣️ Audio Translator",
                "🏛️ Culture & Tradition",
                "✈️ Itinerary Planner",
                "💰 Budget Estimator",
                "💱 Currency Converter",
                "🗺️ Route Planner",
                "🧠 Travel Trivia",
                "⭐ Public Feedback"
            ]
        )

if __name__ == "__main__":
    print("Starting Zenix Travel Companion...")
    print(f"Feedback database file: {FEEDBACK_DB_FILE}")

    # Initialize database on startup
    db = load_feedback_database()
    print(f"Database loaded: {db['metadata']['total_feedback']} feedback entries")

    interface_content.launch(debug=True)
    print("Gradio Interface launched! Access it via the public URL above.")