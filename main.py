import asyncio
from audio import WakeWordListener, PorcupineBuiltinKeyword


async def simple_test():
    """Simple interactive test"""

    print("🎤 Simple Wake Word Test")
    print("=" * 30)

    # Choose wake word
    wake_word = PorcupineBuiltinKeyword.PICOVOICE
    sensitivity = 0.8

    print(f"Wake Word: '{wake_word.value}'")
    print(f"Sensitivity: {sensitivity}")
    print("\nInstructions:")
    print("1. Wait for 'Listening...' message")
    print("2. Say the wake word clearly")
    print("3. Wait for detection or press Ctrl+C to stop")
    print("-" * 30)

    try:
        with WakeWordListener(wakeword=wake_word, sensitivity=sensitivity) as listener:
            print("🔊 Listening... (say the wake word now)")

            detected = await listener.listen_for_wakeword_async()

            if detected:
                print("🎉 SUCCESS! Wake word detected!")
            else:
                print("😴 Listening stopped without detection")

    except KeyboardInterrupt:
        print("\n⏹️  Test stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Test failed")


if __name__ == "__main__":
    asyncio.run(simple_test())
