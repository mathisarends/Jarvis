from agents.realtime import RealtimeSession


async def main():
    async with RealtimeSession(model="gpt-realtime-preview") as session:
        # Mikrofon starten & Input streamen
        await session.start_microphone()

        # Agent antwortet in Sprache -> Lautsprecher
        await session.enable_audio_output()

        # Nutzertext triggern (optional)
        await session.send_user_text("Erzähl mir einen Witz!")

        async for event in session.events():
            print("Event:", event)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
