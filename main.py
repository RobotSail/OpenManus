import asyncio

from app.agent.manus import Manus
from app.agent_thought_logger import cleanup_thought_loggers
from app.logger import logger


async def main():
    # Create and initialize Manus agent
    agent = await Manus.create()
    try:
        prompt = input("Enter your prompt: ")
        if not prompt.strip():
            logger.warning("Empty prompt provided.")
            return

        logger.warning("Processing your request...")
        await agent.run(prompt)
        logger.info("Request processing completed.")
    except KeyboardInterrupt:
        logger.warning("Operation interrupted.")
    finally:
        # Ensure agent resources are cleaned up before exiting
        await agent.cleanup()
        # Clean up thought loggers
        cleanup_thought_loggers()


if __name__ == "__main__":
    asyncio.run(main())
