"""
gemini_utils.py
----------------
All AI / recommendation logic for PocketSmart AI lives here:
- Talking to Google's Gemini model (text + image)
- Building prompts for Home / Party / Jewelry planners
- Parsing Gemini's JSON responses safely
- Attaching "shop this" links for Indian e-commerce platforms
- A simple in-memory history store (per user)
"""

import os
import re
import json
import uuid
import shutil
import urllib.parse
from datetime import datetime
from typing import Optional, Dict, Any, List

from pydantic import BaseModel
from PIL import Image
import google.generativeai as genai

# --------------------------------------------------------------------------
# Gemini configuration
# --------------------------------------------------------------------------

API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError(
        "No Google/Gemini API key found. Please set GOOGLE_API_KEY in your .env file."
    )

genai.configure(api_key=API_KEY)

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
model = genai.GenerativeModel(MODEL_NAME)

USD_TO_INR_RATE = float(os.getenv("USD_TO_INR_RATE", "83.0"))


def usd_to_inr(amount_usd: float, exchange_rate: float = USD_TO_INR_RATE) -> float:
    """Convert USD amount to INR using the specified exchange rate."""
    return amount_usd * exchange_rate


# --------------------------------------------------------------------------
# Pydantic input models (imported by app.py)
# --------------------------------------------------------------------------

class HomeBudgetInput(BaseModel):
    total_budget: float
    num_lights: int = 0
    num_fans: int = 0
    num_furniture: int = 0
    num_dining_tables: int = 0
    has_living_room: bool = False
    has_kitchen: bool = False
    has_bedroom: bool = False
    additional_requirements: Optional[str] = None


class PartyBudgetInput(BaseModel):
    total_budget: float
    num_guests: int
    party_type: str
    venue_type: Optional[str] = None
    needs_catering: bool = True
    needs_decoration: bool = True
    needs_entertainment: bool = True
    additional_requirements: Optional[str] = None


class JewelryBudgetInput(BaseModel):
    total_budget: float
    occasion: str
    preferences: Optional[str] = None


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def extract_json_from_response(text: str) -> dict:
    """
    Gemini sometimes wraps JSON in ```json ... ``` fences or adds stray text.
    This pulls out the first valid JSON object it can find.
    """
    if not text:
        raise ValueError("Empty response from Gemini")

    cleaned = text.strip()
    cleaned = re.sub(r"^```(json)?", "", cleaned.strip(), flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"```$", "", cleaned.strip()).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Fallback: find the outermost { ... } block
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = cleaned[start:end + 1]
        try:
            return json.loads(snippet)
        except json.JSONDecodeError as e:
            raise ValueError(f"Could not parse JSON from Gemini response: {e}\nRaw: {text[:500]}")

    raise ValueError(f"No JSON object found in Gemini response.\nRaw: {text[:500]}")


def save_upload_file(upload_file, upload_dir: str = "static/uploads") -> str:
    """Save an uploaded FastAPI UploadFile to disk and return its path."""
    os.makedirs(upload_dir, exist_ok=True)
    ext = os.path.splitext(upload_file.filename or "")[1] or ".png"
    filename = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}{ext}"
    file_path = os.path.join(upload_dir, filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
    return file_path


def build_shopping_links(search_terms: str, platforms: List[str]) -> Dict[str, str]:
    """Build 'shop this item' links for the given platforms."""
    if not search_terms:
        return {}
    q = urllib.parse.quote_plus(search_terms)
    all_links = {
        "amazon": f"https://www.amazon.in/s?k={q}",
        "flipkart": f"https://www.flipkart.com/search?q={q}",
        "ikea": f"https://www.ikea.com/in/en/search/?q={q}",
        "myntra": f"https://www.myntra.com/search?q={q}",
        "ajio": f"https://www.ajio.com/search/?text={q}",
        "bigbasket": f"https://www.bigbasket.com/ps/?q={q}",
        "swiggy": f"https://www.swiggy.com/search?query={q}",
        "zomato": f"https://www.zomato.com/search?q={q}",
        "bookmyshow": f"https://in.bookmyshow.com/search?q={q}",
        "google": f"https://www.google.com/search?q={q}",
        "booking": f"https://www.booking.com/search.html?ss={q}",
        "makemytrip": f"https://www.makemytrip.com/hotels/hotel-listing/?searchText={q}",
        "oyorooms": f"https://www.oyorooms.com/search/?location={q}",
        "nobroker": f"https://www.nobroker.in/property/search?searchTerm={q}",
        "bluestone": f"https://www.bluestone.com/search.html?query={q}",
        "tanishq": f"https://www.tanishq.co.in/search?q={q}",
        "caratlane": f"https://www.caratlane.com/search?q={q}",
        "melorra": f"https://www.melorra.com/search?q={q}",
        "meesho": f"https://www.meesho.com/search?q={q}",
    }
    return {p: all_links[p] for p in platforms if p in all_links}


CATEGORY_PLATFORMS = {
    "venue": ["google", "booking", "makemytrip", "oyorooms", "nobroker"],
    "catering": ["swiggy", "zomato", "bigbasket", "amazon", "flipkart"],
    "food": ["swiggy", "zomato", "bigbasket", "amazon", "flipkart"],
    "drinks": ["swiggy", "zomato", "bigbasket", "amazon", "flipkart"],
    "decoration": ["amazon", "flipkart", "meesho", "myntra"],
    "entertainment": ["bookmyshow", "amazon", "flipkart"],
    "gifts": ["amazon", "flipkart", "myntra", "meesho"],
    "photography": ["google", "amazon", "flipkart"],
    "music": ["amazon", "flipkart", "bookmyshow"],
    "games": ["amazon", "flipkart"],
    "accessories": ["amazon", "flipkart", "myntra", "meesho"],
    "transportation": ["makemytrip", "google"],
    "return_gifts": ["amazon", "flipkart", "myntra", "meesho"],
    "lighting": ["amazon", "flipkart", "ikea"],
    "fixtures & furniture": ["amazon", "flipkart", "ikea"],
    "furniture": ["amazon", "flipkart", "ikea"],
    "ceiling_fans": ["amazon", "flipkart", "ikea"],
    "dining tables": ["amazon", "flipkart", "ikea"],
}
DEFAULT_PLATFORMS = ["amazon", "flipkart", "google"]
JEWELRY_PLATFORMS = ["amazon", "flipkart", "bluestone", "tanishq", "caratlane", "melorra", "meesho"]


def _attach_links_to_items(items: List[dict], platforms: List[str]):
    for item in items:
        search_terms = item.get("search_terms", "") or item.get("name", "")
        item["shopping_links"] = build_shopping_links(search_terms, platforms)


# --------------------------------------------------------------------------
# 1) HOME INTERIOR PLANNER
# --------------------------------------------------------------------------

def get_home_recommendations(budget_input: HomeBudgetInput) -> dict:
    """Generate home interior recommendations within budget, in INR, for the Indian market."""
    try:
        prompt = f"""
You are a budget-conscious Indian home interior design assistant.

I need interior design product recommendations for a home in India with a total budget of
Rs.{budget_input.total_budget:.2f}.

Requirements:
- {budget_input.num_lights} lights/lighting fixtures
- {budget_input.num_fans} ceiling fans
- {budget_input.num_furniture} furniture pieces
- {budget_input.num_dining_tables} dining tables

Additional rooms to consider:
{"- Living room" if budget_input.has_living_room else ""}
{"- Kitchen" if budget_input.has_kitchen else ""}
{"- Bedroom" if budget_input.has_bedroom else ""}

Additional requirements: {budget_input.additional_requirements or "None"}

Please provide a detailed budget breakdown with product recommendations available in India.
Use Indian brands and pricing. Include short search terms suitable for Indian shopping platforms
(Amazon, Flipkart, IKEA) for every single item.

Respond with ONLY valid JSON (no markdown fences, no commentary) in exactly this structure:
{{
  "total_budget": {budget_input.total_budget:.2f},
  "budget_breakdown": [
    {{
      "category": "lighting",
      "allocation": 0.0,
      "items": [
        {{
          "name": "",
          "description": "",
          "estimated_price": 0.0,
          "quantity": 0,
          "search_terms": ""
        }}
      ]
    }}
  ],
  "remaining_budget": 0.0,
  "additional_suggestions": ["", ""]
}}

Ensure total item costs across all categories stay within the given budget.
""".strip()

        response = model.generate_content(prompt)
        result = extract_json_from_response(response.text)

        # Attach shopping links per item
        for category in result.get("budget_breakdown", []):
            cat_name = str(category.get("category", "")).lower()
            platforms = CATEGORY_PLATFORMS.get(cat_name, DEFAULT_PLATFORMS)
            _attach_links_to_items(category.get("items", []), platforms)

        # Build a calculation table summarizing cost per category
        result["calculation_table"] = []
        for category in result.get("budget_breakdown", []):
            items = category.get("items", [])
            total_cost = sum(
                float(i.get("estimated_price", 0)) * max(int(i.get("quantity", 1) or 1), 1)
                for i in items
            )
            pct = (total_cost / result["total_budget"] * 100) if result.get("total_budget") else 0
            result["calculation_table"].append({
                "category": category.get("category", ""),
                "items_count": len(items),
                "total_cost": round(total_cost, 2),
                "percentage_of_budget": round(pct, 1),
            })

        return result

    except Exception as e:
        raise RuntimeError(f"Error generating home recommendations: {str(e)}")


# --------------------------------------------------------------------------
# 2) PARTY PLANNER
# --------------------------------------------------------------------------

def get_party_recommendations(budget_input: PartyBudgetInput) -> dict:
    """Generate party planning recommendations within budget, in INR, for the Indian market."""
    try:
        prompt = f"""
You are a budget-conscious Indian event/party planning assistant.

I need party planning recommendations for India with a total budget of Rs.{budget_input.total_budget:.2f}.

Party details:
- Type: {budget_input.party_type}
- Number of guests: {budget_input.num_guests}
- Venue type: {budget_input.venue_type or "Not specified"}
- Catering needed: {"Yes" if budget_input.needs_catering else "No"}
- Decoration needed: {"Yes" if budget_input.needs_decoration else "No"}
- Entertainment needed: {"Yes" if budget_input.needs_entertainment else "No"}

Additional requirements: {budget_input.additional_requirements or "None"}

Please provide a detailed budget breakdown with specific recommendations available in India using
INR prices. Use Indian brands, services, and typical cost expectations. Include short search terms
for each item suitable for Indian websites such as Swiggy, Zomato, BookMyShow, Amazon, Flipkart.

Respond with ONLY valid JSON (no markdown fences, no commentary) in exactly this structure:
{{
  "total_budget": {budget_input.total_budget:.2f},
  "budget_breakdown": [
    {{
      "category": "catering",
      "allocation": 0.0,
      "items": [
        {{
          "name": "",
          "description": "",
          "estimated_price": 0.0,
          "quantity": 1,
          "search_terms": ""
        }}
      ]
    }}
  ],
  "venue_suggestions": [
    {{
      "name": "",
      "type": "",
      "capacity": 0,
      "estimated_cost": 0.0,
      "search_terms": ""
    }}
  ],
  "remaining_budget": 0.0,
  "additional_suggestions": ["", ""]
}}

Ensure all costs are in INR and the total does not exceed the given budget.
""".strip()

        response = model.generate_content(prompt)
        result = extract_json_from_response(response.text)

        # Item-level shopping links
        for category in result.get("budget_breakdown", []):
            cat_name = str(category.get("category", "")).lower()
            platforms = CATEGORY_PLATFORMS.get(cat_name, DEFAULT_PLATFORMS)
            _attach_links_to_items(category.get("items", []), platforms)

        # Venue-level search links
        venue_platforms = CATEGORY_PLATFORMS["venue"]
        for venue in result.get("venue_suggestions", []):
            search_terms = venue.get("search_terms", "") or venue.get("name", "")
            venue["search_links"] = build_shopping_links(search_terms, venue_platforms)

        # Calculation table (INR) grouped by category
        result["calculation_table_inr"] = []
        categories: Dict[str, dict] = {}
        for category in result.get("budget_breakdown", []):
            cat_name = category.get("category", "Misc")
            items = category.get("items", [])
            total_cost = sum(
                float(i.get("estimated_price", 0)) * max(int(i.get("quantity", 1) or 1), 1)
                for i in items
            )
            if cat_name not in categories:
                categories[cat_name] = {
                    "category": cat_name,
                    "items_count": 0,
                    "total_cost": 0.0,
                    "percentage_of_budget": 0.0,
                }
            categories[cat_name]["items_count"] += len(items)
            categories[cat_name]["total_cost"] += total_cost

        for cat_data in categories.values():
            if result.get("total_budget"):
                cat_data["percentage_of_budget"] = round(
                    (cat_data["total_cost"] / result["total_budget"]) * 100, 1
                )
            cat_data["total_cost"] = round(cat_data["total_cost"], 2)
            result["calculation_table_inr"].append(cat_data)

        return result

    except Exception as e:
        raise RuntimeError(f"Error generating party recommendations: {str(e)}")


# --------------------------------------------------------------------------
# 3) JEWELRY PLANNER
# --------------------------------------------------------------------------

def get_jewelry_recommendations(budget_input: JewelryBudgetInput, image_path: Optional[str] = None) -> dict:
    """Generate jewelry recommendations based on optional uploaded outfit image and budget (INR)."""
    try:
        base_prompt = f"""
You are a budget-conscious Indian jewelry stylist.

I need jewelry recommendations for India with a total budget of Rs.{budget_input.total_budget:.2f}.

Occasion: {budget_input.occasion}
Preferences: {budget_input.preferences or "Not specified"}
Provide only India-relevant styles, availability, and price ranges in INR.
""".strip()

        if image_path:
            img = Image.open(image_path)
            prompt = base_prompt + """

An image of the outfit is uploaded. Suggest jewelry that complements it, considering color,
design, and occasion appropriateness.

Respond with ONLY valid JSON (no markdown fences, no commentary) in exactly this structure:
{
  "outfit_analysis": {
    "colors": [],
    "style": "",
    "formality": ""
  },
  "total_budget": 0.0,
  "jewelry_recommendations": [
    {
      "item_type": "",
      "description": "",
      "style": "",
      "estimated_price": 0.0,
      "search_terms": ""
    }
  ],
  "remaining_budget": 0.0,
  "styling_tips": []
}

Make sure prices are in INR and stay within budget. Include Indian-friendly search terms for shopping.
"""
            response = model.generate_content([prompt, img])
        else:
            prompt = base_prompt + """

Respond with ONLY valid JSON (no markdown fences, no commentary) in exactly this structure:
{
  "total_budget": 0.0,
  "jewelry_recommendations": [
    {
      "item_type": "",
      "description": "",
      "style": "",
      "estimated_price": 0.0,
      "search_terms": ""
    }
  ],
  "remaining_budget": 0.0,
  "styling_tips": []
}

Keep prices in INR and relevant to Indian brands.
"""
            response = model.generate_content(prompt)

        result = extract_json_from_response(response.text)
        result.setdefault("total_budget", budget_input.total_budget)

        for item in result.get("jewelry_recommendations", []):
            search_terms = item.get("search_terms", "") or item.get("item_type", "")
            item["shopping_links"] = build_shopping_links(search_terms, JEWELRY_PLATFORMS)

        return result

    except Exception as e:
        raise RuntimeError(f"Error generating jewelry recommendations: {str(e)}")


# --------------------------------------------------------------------------
# Simple in-memory recommendation history (per username)
# --------------------------------------------------------------------------

class RecommendationRecord(BaseModel):
    id: str
    timestamp: str
    recommendation_type: str
    input_summary: dict
    result_summary: dict
    full_result: dict


# username -> list of RecommendationRecord
user_recommendations: Dict[str, List[RecommendationRecord]] = {}


def _summarize_input(recommendation_type: str, input_data: dict) -> dict:
    if recommendation_type == "home":
        return {
            "total_budget": input_data.get("total_budget"),
            "rooms": [
                r for r, flag in [
                    ("Living Room", input_data.get("has_living_room")),
                    ("Kitchen", input_data.get("has_kitchen")),
                    ("Bedroom", input_data.get("has_bedroom")),
                ] if flag
            ],
            "lights": input_data.get("num_lights"),
            "fans": input_data.get("num_fans"),
            "furniture": input_data.get("num_furniture"),
        }
    if recommendation_type == "party":
        needs = [
            n for n, flag in [
                ("catering", input_data.get("needs_catering")),
                ("decoration", input_data.get("needs_decoration")),
                ("entertainment", input_data.get("needs_entertainment")),
            ] if flag
        ]
        return {
            "total_budget": input_data.get("total_budget"),
            "party_type": input_data.get("party_type"),
            "guests": input_data.get("num_guests"),
            "needs": needs,
        }
    if recommendation_type == "jewelry":
        return {
            "total_budget": input_data.get("total_budget"),
            "occasion": input_data.get("occasion"),
            "with_outfit_image": bool(input_data.get("image")),
        }
    return input_data


def _summarize_result(recommendation_type: str, result: dict) -> dict:
    return {
        "total_budget": result.get("total_budget"),
        "remaining_budget": result.get("remaining_budget"),
    }


def save_to_history(username: str, recommendation_type: str, input_data: dict, result: dict) -> RecommendationRecord:
    record = RecommendationRecord(
        id=str(uuid.uuid4()),
        timestamp=datetime.utcnow().isoformat(),
        recommendation_type=recommendation_type,
        input_summary=_summarize_input(recommendation_type, input_data),
        result_summary=_summarize_result(recommendation_type, result),
        full_result=result,
    )
    user_recommendations.setdefault(username, []).append(record)
    return record
