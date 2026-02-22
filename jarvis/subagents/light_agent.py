from typing import Annotated

from rtvoice import SubAgent, Tools
from llmify import ChatOpenAI

from hueify import Room, RoomLookup


class LightAgent(SubAgent):
    def __init__(self) -> None:
        super().__init__(
            name="Light Agent",
            description="Controls Philips Hue lights, rooms, and zones.",
            instructions=(
                "You control smart lights. "
                "Always call list_entities first to know which rooms and zones are available. "
                "Then act on the most appropriate one based on the user's request. "
                "Confirm what you did in a short, natural sentence."
            ),
            tools=self._build_tools(),
            llm=ChatOpenAI(model="gpt-4o-mini", temperature=0.2),
            handoff_instructions="Use this agent for light and ambiance control.",
            pending_message="Adjusting the lights...",
        )

    def _build_tools(self) -> Tools:
        tools = Tools()

        @tools.action("List all available rooms and zones so you know what to control.")
        async def list_entities() -> str:
            room_lookup = RoomLookup()

            rooms = await room_lookup.get_all_entities()

            return (
                f"Rooms: {', '.join(r.name for r in rooms)}\n"
            )
        
        @tools.action("Turn on all lights in a room.")
        async def turn_on_room(room_name: Annotated[str, "Name of the room"]) -> str:
            room = await Room.from_name(room_name)
            result = await room.turn_on()
            return str(result)

        @tools.action("Turn off all lights in a room.")
        async def turn_off_room(room_name: Annotated[str, "Name of the room"]) -> str:
            room = await Room.from_name(room_name)
            result = await room.turn_off()
            return str(result)
        
        @tools.action("Decrease the brightness of an entire room by a percentage (0–100).")
        async def decrease_room_brightness(
            room_name: Annotated[str, "Name of the room"],
            percentage: Annotated[float, "Brightness percentage (0–100)"],
        ) -> str:
            room = await Room.from_name(room_name)
            result = await room.decrease_brightness_percentage(percentage)
            return str(result)
        
        @tools.action("Increase the brightness of an entire room by a percentage (0–100).")
        async def increase_room_brightness(
            room_name: Annotated[str, "Name of the room"],
            percentage: Annotated[float, "Brightness percentage (0–100)"],
        ) -> str:
            room = await Room.from_name(room_name)
            result = await room.increase_brightness_percentage(percentage)
            return str(result)

        @tools.action("Activate a named scene in a room.")
        async def activate_room_scene(
            room_name: Annotated[str, "Name of the room"],
            scene_name: Annotated[str, "Name of the scene to activate"],
        ) -> str:
            room = await Room.from_name(room_name)
            result = await room.activate_scene(scene_name)
            return str(result)
        
        @tools.action("Get all available scenes for a specific room.")
        async def get_room_scenes(
            room_name: Annotated[str, "Name of the room"],
        ) -> str:
            room = await Room.from_name(room_name)
            scenes = await room.get_scenes()
            if not scenes:
                return f"No scenes found for room '{room_name}'."
            return f"Scenes in {room_name}: {', '.join(s.name for s in scenes)}"

        @tools.action("Get the currently active scene in a room.")
        async def get_active_room_scene(
            room_name: Annotated[str, "Name of the room"],
        ) -> str:
            room = await Room.from_name(room_name)
            try:
                scene = await room.get_active_scene()
                return f"Active scene in {room_name}: '{scene.name}'"
            except Exception as e:
                return f"No active scene in {room_name}: {e}"
            
        return tools
    
    