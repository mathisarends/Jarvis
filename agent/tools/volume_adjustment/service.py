from agents import Agent, Runner

from agent.tools.volume_adjustment.models import ResponseSpeedAdjustment


async def run_volume_adjustment_agent(
    instruction: str, current_response_speed: float
) -> ResponseSpeedAdjustment:
    MIN_RATE = 0.25
    MAX_RATE = 1.5

    agent = Agent(
        name="ResponseSpeedAgent",
        model="gpt-5-mini",
        instructions=(
            "You are a response speed adjustment assistant.\n\n"
            f"Current speed: {current_response_speed:.3f}\n"
            f"Allowed range: [{MIN_RATE}, {MAX_RATE}]\n\n"
            "RULES:\n"
            "1) Interpret commands like:\n"
            "   - 'faster', 'speed up', 'increase' => +10%\n"
            "   - 'slower', 'slow down', 'decrease' => -10%\n"
            "   - 'much faster' => +25%, 'much slower' => -25%\n"
            "2) If an explicit percentage is provided:\n"
            "   - 'increase by X%' or 'decrease by X%' => apply ±X% to the current speed.\n"
            "   - 'set to X%' => set absolute speed to X/100 (e.g., 125% => 1.25).\n"
            "3) Always clamp the final result to the allowed range.\n"
            "4) Respond only with valid structured JSON matching the schema.\n\n"
            "OUTPUT:\n"
            "- new_response_speed: final float multiplier (0.25–1.5)"
        ),
        output_type=ResponseSpeedAdjustment,
    )

    result = await Runner.run(agent, instruction)
    return result.final_output
