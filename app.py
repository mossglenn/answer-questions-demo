import gradio as gr
import ast
import random
import copy
import re
from huggingface_hub import InferenceClient
from openai import AsyncOpenAI, OpenAIError


CONFIG = {
    "show_placeholder_text": True,
    "local_only": False,
    "demo_text_status": (
        "This is placeholder text for demo only. If the program has notes "
        "attached to this question in the copy repository, "
        "it would appear here."
    ),
    "demo_text_alert": (
        "This is placeholder text for demo only. To get a real result, try"
        "including one or more of the following phrases in your answer:"
        "<ul><li>research subject</li><li>human subjects</li></ul>"
    ),
    "demo_text_suggestions": (
        "This is placeholder text for demo only. If the app is connected to "
        "OpenAI, ChatGPT will write suggestions for how to rewrite the "
        "learner's answer to be more like the accapted answer."
    ),
    "forbidden_phrases": [
        'research subject', 'research subjects',
        'human subject', 'human subjects'
    ],
    "initial_state": {
        "question": "",
        "answer": "",
        "status": "",
        "attempt": "",
        "alert_text": "",
        "score_label": "Good Score!",
        "score": 66,
        "suggestions": "",
        "showScore": False,
        "showStatus": False,
        "showPhrasingAlert": False,
        "showSuggestions": False,
    },
}

try:
    client = InferenceClient()
except Exception as e:
    print(f"Error initializing inference client: {e}")
    client = None

# Load FAQ and select a random entry
with open('faq_dict.txt', 'r') as fd:
    faq = ast.literal_eval(fd.read())
print('> FAQ loaded...')


async def fetchSuggestions(state):
    """Uses OpenAI API to summarize differences between two text blocks"""
    print('> Requesting summary of differences...')
    openAIclient = AsyncOpenAI()
    prompt = (
        f"Write a short paragraph summarizing the one or two most "
        f"important ways that the submitted answer block of text below "
        f"can be improved to become more similar to the approved answer "
        f"block of text. Do not include a new version of the submitted "
        f"answer. Do not include any lists. The summary text should "
        f"start with 'To improve your answer, think about ', "
        f"Approved answer: {state['answer']} "
        f"Submitted answer: {state['attempt']}"
    )
    try:
        completion = await openAIclient.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        print(">>>>> Summarized...")
        return f"{completion.choices[0].message.content}"

    except OpenAIError as e:
        print(f"Error: {e}")
        return (
            "Sorry, an error occured while trying to generate a suggestion "
            "for improvements to your answer."
        )


def getScoreLabel(score: int) -> str:
    if score > 79:
        return "Very High"
    elif score > 59:
        return "High"
    elif score > 39:
        return "Moderate"
    elif score > 19:
        return "Low"
    else:
        return "Very Low"


def dump(state):
    print("***** Current Values in State *****")
    if isinstance(state, dict):
        max_key_length = max(len(key) for key in state.keys())
        for key, value in state.items():
            print(f"{key.rjust(max_key_length)} : {value}")
        print("******** End of State ********")
    else:
        print("Error: 'state' must be a dictionary.")


def printQuestion(state):
    return f"<span class='question-text'>{state['question']}</span>"


def printScoreCard(state):
    raw_score = client.sentence_similarity(
        sentence=state["answer"],
        other_sentences=[state["attempt"]],
        model="sentence-transformers/all-MiniLM-L6-v2"
    )
    int_score = int(raw_score[0] * 100)
    if int_score < 0:
        state["score"] = 0
    else:
        state["score"] = int(raw_score[0] * 100)

    state["score_label"] = getScoreLabel(state["score"])
    state["showScore"] = True

    return (
        f"<div class='center-box'>"
        f"<div role='progressbar' aria-valuenow='{state['score']}' "
        f"aria-valuemin='0' aria-valuemax='100' "
        f"style='--value: {state['score']}'>"
        f"<div class='label'>{state['score_label']}</div></div></div>"
    )


def printForbiddenBox(state):
    # list comprehension
    forbidden_matches: list = [
        phrase
        for phrase in CONFIG['forbidden_phrases']
        if re.search(rf"\b{re.escape(phrase)}\b", state['attempt'].lower())
    ]
    if not forbidden_matches:
        #  include demo text
        if CONFIG["show_placeholder_text"]:
            state['showPhrasingAlert'] = True
            return (
                f"<div class='feedback-card red-card'>"
                f"<div class='feedback-head'>Demo Alert</div>"
                f"{CONFIG['demo_text_alert']}"
                f"</div>"
            )
        else:
            return ""

    state['showPhrasingAlert'] = True
    alert_message = ""

    if len(forbidden_matches) == 1:
        state['showPhrasingAlert'] = True
        alert_message = (
            f"OOPS! You used the phrase <em>'{forbidden_matches[0]}'</em> "
            f"which is forbidden."
            )
    else:
        phrase_list = "<ul>"
        for phrase in forbidden_matches:
            phrase_list += f"<li>{phrase}</li>"
        phrase_list += "</ul>"

        alert_message = (
            f"<p>OH NO! You used these forbidden phrases: {phrase_list}"
            f"<p>The program IRB forbids the use of these phrases!"
        )
    return (
        f"<div class='feedback-card red-card'>"
        f"<div class='feedback-head'>Watch your phrasing!</div>"
        f"{alert_message}"
        f"</div>"
    )


def printStatusBox(state):
    if state["status"]:
        state["showStatus"] = True
    elif CONFIG["show_placeholder_text"]:
        state["status"] = CONFIG["demo_text_status"]
        state["showStatus"] = True

    if state["showStatus"]:
        return (
            f"<div class='feedback-card blue-card'>"
            f"<div class='feedback-head'>Approved Answer Notes:</div>"
            f"{state['status']}</div>"
        )
    else:
        return ""


async def printSuggestionsBox(state):
    if CONFIG["local_only"]:
        suggestion_message = CONFIG["demo_text_suggestions"]
        state["showSuggestions"] = True
    else:
        state["suggestions"] = await fetchSuggestions(state)
        if state["suggestions"]:
            state["showSuggestions"] = True
            suggestion_message = state["suggestions"]
        elif CONFIG["show_placeholder_text"]:
            state["showSuggestions"] = True
            suggestion_message = CONFIG["demo_text_suggestions"]

    if state["showSuggestions"]:
        return (
            f"<div class='feedback-card green-card'>"
            f"<div class='feedback-head'>Suggestions for improvement:</div>"
            f"{suggestion_message}</div>"
        )
    else:
        return ""


def updateEntry(entry, state):
    defaults = [("question", "Question not found"),
                ("answer", "Answer not found"),
                ("status", "Status not found")]
    for key, default in defaults:
        state[key] = entry.get(key, default)

    if (
        CONFIG["show_placeholder_text"]
        and state["status"] == defaults[2][1]
    ):
        state["status"] = CONFIG["demo_text_status"]
    return state


def selectQuestion():
    entry_id = random.choice(list(faq.keys()))
    return faq[entry_id]


def initialize_state_values():
    print('> Initializing...')
    state = copy.deepcopy(CONFIG["initial_state"])
    state = updateEntry(selectQuestion(), state)
    dump(state)
    return state


async def submit(attempt: str, state: dict):
    state['attempt'] = attempt
    return (
        state,
        gr.update(  # attempt_box
            interactive=False,
            lines=1,
            elem_classes=["box-without-border"]
        ),
        gr.update(  # submit_button
            interactive=False,
            variant='secondary'
        ),
        gr.update(  # reset_button
            interactive=True,
            variant='primary'
        ),
        gr.update(  # answer_box
            value=state['answer'],
            elem_classes=["approved-answer", "box-without-border"]
        ),
        gr.update(  # status_box
            value=printStatusBox(state),
            visible=state['showStatus']
        ),
        gr.update(  # forbidden_box
            value=printForbiddenBox(state),
            visible=state['showPhrasingAlert']
        ),
        gr.update(  # score_box
            value=printScoreCard(state),
            visible=state["showScore"]
        ),
        gr.update(  # suggestion_box
            value=await printSuggestionsBox(state),
            visible=state['showSuggestions']
        )
    )


def reset_question(state: dict):
    state = updateEntry(
        selectQuestion(),
        copy.deepcopy(CONFIG["initial_state"])
    )
    return (
        state,
        gr.update(value=printQuestion(state)),  # question_box
        gr.update(  # attempt_box
            value="",
            interactive=True,
            lines=3,
            elem_classes=["box-with-border"]),
        gr.update(  # Hide answer_box
            value="",
            elem_classes=[
                "box-without-border",
                "approved-answer",
                "hidden-box"
            ]
        ),
        gr.update(visible=False),  # Hide status_box
        gr.update(visible=False),  # Hide forbidden_box
        gr.update(visible=False),  # Hide score_box
        gr.update(visible=False),   # Hide suggestion_box
        gr.update(interactive=True, variant='primary'),  # submit_button
        gr.update(interactive=False, variant='secondary'),  # reset_button
    )


with gr.Blocks(css_paths="customStyles.css") as demo:
    state = gr.State(value=initialize_state_values())

    with gr.Row(elem_classes=['title-row']):
        gr.HTML("<div class='title-text'>Answering Questions Demo: </div>"
                "<div class='subtitle-text'>A learning activity for gaining "
                "<b>fluency</b> and <b>confidence</b> when responding to "
                "questions from participants</div>"
                )
    
    with gr.Row(elem_classes=['heading-row']):
        with gr.Column(elem_classes=['heading-row-2']):
            gr.HTML("<div class='instructions-text'>"
                    "<h4>Instructions for Learners</h4>"
                    "<p>A question is randomly chosen from an official FAQ "
                    "and the learner writes their response in the textbox "
                    "below the question. After the submit button is clicked, "
                    "the learner's answer is evaluated and several types of "
                    "feedback are provided.</p>"
                    
                    )
        with gr.Column(elem_classes=['heading-row-1']):
            gr.HTML("<div class='tester-text'>"
                    "<h4>Instructions for Demo Testers</h4>"
                    "<p>Check out the bottom of this page to read "
                    "<a href='#discussion'>How does this work?</a>"
                    "<div class='nota-bena'><p><b>Please Note</b>: This demo "
                    "runs on minimal (mostly free) resources.<br />It may "
                    "take several seconds ⏱️ to respond to a submission with "
                    "feedback. </div>"
                    )
 
    with gr.Row():
        gr.HTML("<div class='blocker-row'></div>")

    with gr.Row():
        question_box = gr.HTML(
            value="<div class='initial-text'>Selecting a question...</div>"
        )

    with gr.Row():
        attempt_box = gr.Textbox(
            label="Your answer:",
            lines=7,
            elem_classes=["box-with-border"]
        )

        answer_box = gr.Textbox(
            label="Approved Answer",
            interactive=False,
            elem_classes=[
                "hidden-box",
                "approved-answer",
                "box-without-border"
            ],
            visible=True
        )

    with gr.Row():  # status row
        with gr.Column():
            submit_button = gr.Button(
                value="Submit Your Answer",
                variant="primary"
            )
            reset_button = gr.Button(
                value="Get a New Question",
                variant="secondary",
                interactive=False
            )
        with gr.Column():
            status_box = gr.HTML(visible=False)

    with gr.Row():  # feedback row
        with gr.Column(elem_classes=["center-box"]):
            score_box = gr.HTML(visible=False)
        with gr.Column():
            forbidden_box = gr.HTML(visible=False)
            suggestion_box = gr.HTML(visible=False)

    gr.HTML(
            "<div id='discussion'>"
            "<div style='line-height: 150%;'><h2>How Does This Work?</h2><p>"
            "This is a proof-of-concept demo for a learning activity. "
            "A question is randomly "
            "chosen from an official FAQ and the learner writes their response"
            " in the textbox below the question. After the submit button is "
            "clicked, the learner's answer is evaluated and several types of "
            "feedback are provided:</p>"
            "<p><ul><li><em>Approved Answer</em>: the IRB-approved response to "
            "this question on the official FAQ</li>"
            "<li><em>Approved Answer Notes:</em> notes taken directly from the "
            "official Copy Respository (not all answers have notes)</li>"
            "<li><em>Score</em>: Appearing below the learner's answer, the "
            "score consists of a percentage and a interpretation. The "
            "percentage is based on the similarity score resulting from the "
            "Semantic Textual Similarity task in the HuggingFace InferenceAPI."
            "The interpretive label below the score were created for this "
            "demo.</li>"
            "<li><em>Watch Your Phrasing Alert:</em> regex is used to match "
            "strings to the learner's answer. The strings are an unofficial "
            "list of words and phrases that the program has asked staff to "
            "avoid (the demo list includes only 'research subjects' and "
            "'human subjects'). The alert box appears when a match is found "
            "and the matching strings are listed.</li>"
            "<li><em>Suiggestions for Improvement</em>: the results of a "
            "call to OpenAI API that uses a ChatGPT model to generate a "
            "paragraph describing ways the learner's answer could be changed "
            "to become semantically closer to the approved answer.</li>"
            "</ul></p></div>"
            "</div>"
            )

    demo.load(
        fn=printQuestion,
        inputs=state,
        outputs=question_box
    )

    # Event bindings
    submit_button.click(
        fn=submit,
        inputs=[attempt_box, state],
        outputs=[
            state,
            attempt_box,
            submit_button,
            reset_button,
            answer_box,
            status_box,
            forbidden_box,
            score_box,
            suggestion_box
        ]
    )

    reset_button.click(
        fn=reset_question,
        inputs=state,
        outputs=[
            state,
            question_box,
            attempt_box,
            answer_box,
            status_box,
            forbidden_box,
            score_box,
            suggestion_box,
            submit_button,
            reset_button
        ]
    )

demo.launch(allowed_paths=['information-svgrepo-com.png'])
