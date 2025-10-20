"""
Live Cricket Score Tracking using Cricbuzz
Tracks match situations and triggers alerts by scraping Cricbuzz
"""
import logging
import re
import httpx
from typing import Dict, Optional, List
from datetime import datetime
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


async def get_live_matches_from_cricbuzz() -> List[Dict]:
    """
    Fetch all live matches from Cricbuzz homepage
    
    Returns:
        List of live match dictionaries
    """
    try:
        url = "https://www.cricbuzz.com/cricket-match/live-scores"
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        }
        
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        live_matches = []
        
        # Find all live match cards
        match_cards = soup.find_all('div', class_=re.compile(r'cb-mtch-lst.*'))
        
        for card in match_cards:
            try:
                match_info = _parse_cricbuzz_match_card(card)
                if match_info and match_info.get('status') == 'live':
                    live_matches.append(match_info)
                    logger.info(f"🏏 Found live match: {match_info['match']}")
            except Exception as e:
                logger.debug(f"Error parsing match card: {e}")
                continue
        
        return live_matches
        
    except Exception as e:
        logger.error(f"Error fetching from Cricbuzz: {e}")
        return []


async def get_live_cricket_score(team_name: str = "India") -> Optional[Dict]:
    """
    Get live cricket score for a specific team from Cricbuzz
    
    Args:
        team_name: Team name (e.g., "India", "Bangladesh", "Australia")
    
    Returns:
        Dict with match info or None if no live match
        {
            "match": "Bangladesh vs West Indies",
            "score": "93/3 (28.2 ov)",
            "overs": 28.2,
            "overs_remaining": 21.8,
            "is_last_over": False,
            "is_death_overs": False,
            "status": "live",
            "batting_team": "Bangladesh"
        }
    """
    try:
        live_matches = await get_live_matches_from_cricbuzz()
        
        # Find match involving the team
        team_lower = team_name.lower()
        for match in live_matches:
            match_name = match.get('match', '').lower()
            if team_lower in match_name:
                logger.info(f"🏏 Live Match Found: {match['match']}")
                logger.info(f"   Score: {match.get('score', 'N/A')}")
                return match
        
        logger.info(f"No live match found for {team_name}")
        return None
        
    except Exception as e:
        logger.error(f"Error getting live score for {team_name}: {e}")
        return None


def _parse_cricbuzz_match_card(card) -> Optional[Dict]:
    """Parse a Cricbuzz match card to extract match info"""
    try:
        # Get match title (teams)
        title_elem = card.find(['h3', 'a'], class_=re.compile(r'.*text-hvr-underline.*'))
        if not title_elem:
            return None
        
        match_title = title_elem.get_text(strip=True)
        
        # Get score text
        score_elem = card.find('div', class_=re.compile(r'.*cb-scr-wll-chvrn.*'))
        if not score_elem:
            return None
        
        score_text = score_elem.get_text(strip=True)
        
        # Get status
        status_elem = card.find('div', class_=re.compile(r'.*cb-text-live.*|.*cb-text-complete.*'))
        status = "live" if status_elem and "live" in status_elem.get_text(strip=True).lower() else "completed"
        
        # Parse score: "93/3 (28.2)" or "93/3"
        overs = 0.0
        overs_remaining = 50.0
        is_last_over = False
        is_death_overs = False
        
        # Extract overs from score text
        overs_match = re.search(r'\((\d+\.?\d*)\s*(?:ov|overs)?\)', score_text)
        if overs_match:
            overs = float(overs_match.group(1))
            overs_remaining = max(0, 50.0 - overs)
            is_last_over = overs_remaining < 1.0
            is_death_overs = overs_remaining <= 5.0
        
        return {
            "match": match_title,
            "score": score_text,
            "overs": overs,
            "overs_remaining": overs_remaining,
            "is_last_over": is_last_over,
            "is_death_overs": is_death_overs,
            "status": status,
            "raw_html": str(card)[:200]  # For debugging
        }
        
    except Exception as e:
        logger.debug(f"Error parsing match card: {e}")
        return None


def _parse_live_score(title: str, snippet: str, team_name: str) -> Optional[Dict]:
    """
    Parse live cricket score from Google search result
    
    Examples:
    - "India 245/8 (49.2 ov)"
    - "IND 245-8 in 49.2 overs"
    - "India need 51 runs from 5 balls"
    """
    text = f"{title} {snippet}".lower()
    
    # Check if it's a live match
    if "live" not in text and "vs" not in text:
        return None
    
    match_info = {
        "match": "",
        "score": "",
        "overs": 0.0,
        "overs_remaining": 0.0,
        "is_last_over": False,
        "status": "live",
        "target": None,
        "runs_needed": None,
    }
    
    # Extract match name (e.g., "India vs Australia")
    vs_pattern = r"([\w\s]+)\s+vs\s+([\w\s]+)"
    vs_match = re.search(vs_pattern, text, re.IGNORECASE)
    if vs_match:
        match_info["match"] = f"{vs_match.group(1).strip()} vs {vs_match.group(2).strip()}"
    
    # Extract score: "245/8" or "245-8"
    score_pattern = r"(\d+)[/-](\d+)"
    score_match = re.search(score_pattern, text)
    if score_match:
        runs = score_match.group(1)
        wickets = score_match.group(2)
        match_info["score"] = f"{runs}/{wickets}"
    
    # Extract overs: "49.2 ov" or "49.2 overs"
    overs_pattern = r"(\d+\.?\d*)\s*(?:ov|overs)"
    overs_match = re.search(overs_pattern, text)
    if overs_match:
        overs = float(overs_match.group(1))
        match_info["overs"] = overs
        
        # Calculate remaining overs (assuming 50 overs match)
        total_overs = 50.0
        match_info["overs_remaining"] = total_overs - overs
        
        # Check if last over (< 1 over remaining)
        match_info["is_last_over"] = match_info["overs_remaining"] < 1.0
    
    # Extract runs needed: "need 51 runs"
    need_pattern = r"need\s+(\d+)\s+runs?"
    need_match = re.search(need_pattern, text)
    if need_match:
        match_info["runs_needed"] = int(need_match.group(1))
    
    # Extract target: "target 296" or "chasing 296"
    target_pattern = r"(?:target|chasing)\s+(\d+)"
    target_match = re.search(target_pattern, text)
    if target_match:
        match_info["target"] = int(target_match.group(1))
    
    # Only return if we found meaningful info
    if match_info["match"] and (match_info["score"] or match_info["overs"] > 0):
        return match_info
    
    return None


async def check_match_alert_conditions(team_name: str, user_conditions: Dict) -> Optional[Dict]:
    """
    Check if current match state matches user's alert conditions
    
    Args:
        team_name: Team to track
        user_conditions: User's alert conditions
            {
                "last_over": True,
                "last_5_overs": True,
                "runs_needed_under": 50,
                "wickets_remaining": 2
            }
    
    Returns:
        Match info if conditions met, None otherwise
    """
    match_info = await get_live_cricket_score(team_name)
    
    if not match_info:
        return None
    
    # Check user conditions
    conditions_met = []
    
    # Last over condition
    if user_conditions.get("last_over") and match_info.get("is_last_over"):
        conditions_met.append("Last over of innings")
    
    # Last 5 overs condition  
    if user_conditions.get("last_5_overs") and match_info.get("overs_remaining", 0) <= 5:
        conditions_met.append(f"Last {match_info['overs_remaining']:.1f} overs remaining")
    
    # Runs needed condition
    if user_conditions.get("runs_needed_under"):
        threshold = user_conditions["runs_needed_under"]
        if match_info.get("runs_needed") and match_info["runs_needed"] <= threshold:
            conditions_met.append(f"Need only {match_info['runs_needed']} runs")
    
    if conditions_met:
        match_info["alert_reasons"] = conditions_met
        return match_info
    
    return None


# Simple direct scraping method (backup if API quota exhausted)
async def get_live_score_simple(team_name: str = "India") -> Optional[str]:
    """
    Simple method: Just scrape Google search results page HTML
    More reliable but requires parsing HTML
    """
    try:
        query = f"{team_name} cricket live score"
        url = f"https://www.google.com/search?q={query}"
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        }
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url, headers=headers)
            html = response.text
        
        # Look for score in HTML (Google shows it in structured data)
        # Pattern: <div class="imso_mh__score">245/8</div>
        score_pattern = r'class="imso_mh__score[^"]*">([^<]+)</div>'
        score_match = re.search(score_pattern, html)
        
        if score_match:
            score = score_match.group(1).strip()
            logger.info(f"🏏 Live Score: {score}")
            return score
        
        return None
        
    except Exception as e:
        logger.error(f"Error scraping live score: {e}")
        return None


# Test function
async def test_live_score(team_name: str = "India"):
    """Test live score tracking"""
    print(f"\n🏏 Testing Live Cricket Score Tracking for {team_name}\n")
    
    # Test 1: Fetch all live matches from Cricbuzz
    print(f"1. Fetching all live matches from Cricbuzz...")
    live_matches = await get_live_matches_from_cricbuzz()
    
    print(f"   Found {len(live_matches)} live matches:")
    for i, match in enumerate(live_matches, 1):
        print(f"   {i}. {match['match']} - {match['score']}")
    
    # Test 2: Get specific team match
    print(f"\n2. Looking for {team_name} match...")
    match_info = await get_live_cricket_score(team_name)
    
    if match_info:
        print(f"   ✅ Match: {match_info['match']}")
        print(f"   Score: {match_info['score']}")
        print(f"   Overs: {match_info['overs']}")
        print(f"   Remaining: {match_info['overs_remaining']:.1f} overs")
        print(f"   Last Over: {match_info['is_last_over']}")
        if match_info.get('runs_needed'):
            print(f"   Runs Needed: {match_info['runs_needed']}")
    else:
        print("   ℹ️  No live match found for this team")
    
    # Test 3: Check alert conditions
    if match_info:
        print("\n3. Checking alert conditions...")
        conditions = {
            "last_over": True,
            "last_5_overs": True,
            "runs_needed_under": 100
        }
        
        alert = await check_match_alert_conditions(team_name, conditions)
        if alert:
            print(f"   🚨 ALERT! Conditions met:")
            for reason in alert.get('alert_reasons', []):
                print(f"      - {reason}")
        else:
            print("   ℹ️  No alerts triggered (conditions not met)")


if __name__ == "__main__":
    import asyncio
    import sys
    
    # Allow team name as command line argument
    team = sys.argv[1] if len(sys.argv) > 1 else "India"
    asyncio.run(test_live_score(team))
