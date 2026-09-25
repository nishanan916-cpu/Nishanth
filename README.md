# PocketSmart AI — Smart Budget & Recommendation Assistant

An AI-powered budget planner covering three domains: **Home Interior**, **Party Planning**,
and **Jewelry Recommendations**. Built with **FastAPI** + **Google Gemini** + **Jinja2/HTML/CSS**.

---

## 1. Project structure

```
pocketsmart_ai/
├── app.py                 # FastAPI app: routes, auth, sessions
├── gemini_utils.py         # Gemini prompts, JSON parsing, shopping links, history
├── requirements.txt
├── .env.example             # copy to .env and fill in your Gemini key
├── static/
│   ├── styles.css
│   ├── script.js
│   └── uploads/            # uploaded outfit images land here
└── templates/
    ├── base.html
    ├── index.html
    ├── login.html
    ├── register.html
    ├── dashboard.html
    ├── home_planner.html
    ├── party_planner.html
    ├── jewelry_planner.html
    └── history.html
```

> **Note on data storage:** to keep this project simple to run, user accounts, sessions, and
> recommendation history are stored **in memory** (they reset when you restart the server).
> For production use, swap the dictionaries in `app.py` / `gemini_utils.py` for a real database
> (PostgreSQL, SQLite, MongoDB, etc.).

---

## 2. Prerequisites

- Python 3.10+ installed
- A free Google Gemini API key: https://aistudio.google.com/app/apikey
- VS Code (recommended) with the Python extension

---

## 3. Setup in VS Code

1. **Unzip** the project and open the folder in VS Code:
   `File → Open Folder... → pocketsmart_ai`

2. **Open a terminal** in VS Code (`` Ctrl+` `` / `` Cmd+` ``).

3. **Create a virtual environment:**

   Windows:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

   macOS / Linux:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Configure your environment variables:**
   - Copy `.env.example` to a new file named `.env`
   - Open `.env` and paste your Gemini API key into `GOOGLE_API_KEY`
   - (Optional) change `SECRET_KEY` to any random string

---

## 4. Running the app

From the project root (with the virtual environment activated):

```bash
uvicorn app:app --reload
```

or simply:

```bash
python app.py
```

Then open your browser at:

```
http://127.0.0.1:8000
```

You should see the PocketSmart AI landing page.

---

## 5. Testing the application

1. Click **Get Started** → **Create Account**, register a user, then **Sign In**.
2. From the **Dashboard**, try each planner:
   - **Home Budget Planner** — enter a budget + room/furniture counts → *Generate Recommendations*
   - **Party Budget Planner** — enter budget, guest count, event type → *Generate Budget Plan*
   - **Jewelry Budget Planner** — enter budget + occasion, optionally upload an outfit photo → *Get Recommendations*
3. Check the **History** page to see saved recommendations and view full details.
4. You can also explore the auto-generated API docs at:
   ```
   http://127.0.0.1:8000/docs
   ```

---

## 6. Common issues

| Problem | Fix |
|---|---|
| `ValueError: No Google/Gemini API key found` | Make sure you created `.env` (not just `.env.example`) and set `GOOGLE_API_KEY`. |
| `ModuleNotFoundError` | Re-run `pip install -r requirements.txt` inside the activated virtual environment. |
| Gemini returns unparseable output / errors | The app already strips markdown fences and retries JSON extraction; if a model name is deprecated, change `GEMINI_MODEL` in `.env` (e.g. to `gemini-2.0-flash`). |
| Login redirects back to login page | Cookies are used for auth — make sure you're testing on `http://127.0.0.1:8000` (not a mixed http/https setup), and that cookies aren't blocked. |
| Port already in use | Run `uvicorn app:app --reload --port 8001` and open that port instead. |

---

## 7. Tech stack

- **Backend:** FastAPI, Uvicorn, python-jose (JWT), Passlib (bcrypt password hashing)
- **AI:** Google Generative AI SDK (`google-generativeai`) using a Gemini text+vision model
- **Frontend:** Jinja2 templates, vanilla HTML/CSS/JS (fetch API)
- **Shopping links:** dynamically built search URLs for Amazon, Flipkart, IKEA, Swiggy, Zomato,
  BookMyShow, OYO Rooms, Tanishq, CaratLane, and more — based on AI-suggested search terms.
