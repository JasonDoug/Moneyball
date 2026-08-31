"""
Stage 3: Contextual & Environmental Feature Engineering
Calculates Park Factors, Weather Vectors (temperature, humidity, wind vector),
Home-Field Advantage, Rest Days, and Travel/Time Zone shifts.
"""

import math
from typing import Dict
from mlb_engine.config import PARK_FACTORS, WEATHER_BASELINES

class ContextFeatureEngineer:
    """Engineers environmental and schedule context features."""

    @staticmethod
    def calculate_environmental_context(venue: str, weather: Dict) -> Dict[str, float]:
        """
        Calculates park factor multiplier and physics-based weather adjustment vector.
        - Higher temp -> lower air density -> more ball carry (+1% total runs per +5 deg F over baseline).
        - Wind blowing out -> increases HR and scoring.
        """
        park_factor = PARK_FACTORS.get(venue, 1.00)
        temp = weather.get("temperature", WEATHER_BASELINES["temp_ref"])
        humidity = weather.get("humidity", WEATHER_BASELINES["humidity_ref"])
        wind_speed = weather.get("wind_speed", WEATHER_BASELINES["wind_ref"])
        wind_dir = weather.get("wind_direction", "Calm")

        # Temp effect: baseline 72F
        temp_delta = temp - WEATHER_BASELINES["temp_ref"]
        temp_multiplier = 1.0 + (temp_delta / 5.0) * 0.01  # +1% per 5 degrees

        # Wind vector effect:
        wind_multiplier = 1.0
        if "Out" in wind_dir:
            wind_multiplier += (wind_speed / 10.0) * 0.04  # +4% per 10mph wind out
        elif "In" in wind_dir:
            wind_multiplier -= (wind_speed / 10.0) * 0.04  # -4% per 10mph wind in

        total_environmental_run_multiplier = park_factor * temp_multiplier * wind_multiplier

        return {
            "park_factor": float(park_factor),
            "temperature_f": float(temp),
            "humidity_pct": float(humidity),
            "wind_speed_mph": float(wind_speed),
            "temp_multiplier": float(temp_multiplier),
            "wind_multiplier": float(wind_multiplier),
            "total_environmental_run_multiplier": float(total_environmental_run_multiplier)
        }

    @staticmethod
    def calculate_schedule_travel_context(
        is_home: bool,
        rest_days: int,
        tz_changes: int
    ) -> Dict[str, float]:
        """Calculates home field advantage, rest, and fatigue penalty due to travel across timezones."""
        home_advantage = 0.04 if is_home else -0.04  # ~4% boost to home win probability
        travel_fatigue = (tz_changes * 0.015) if not is_home else 0.0
        rest_advantage = (rest_days - 1) * 0.01  # Rest advantage

        net_context_boost = home_advantage + rest_advantage - travel_fatigue

        return {
            "is_home": 1.0 if is_home else 0.0,
            "rest_days": float(rest_days),
            "tz_changes": float(tz_changes),
            "net_context_boost": float(net_context_boost)
        }
