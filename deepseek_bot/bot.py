"""
DeepSeek Discord Bot
---------------------

This script implements an autonomous Discord bot powered by the
`deepseek/deepseek-chat-v3-0324:free` language model through the
OpenRouter API.  The bot can converse, summarise discussions,
generate random facts, and perform basic moderation tasks.  It
includes a scheduled daily summary and sets its avatar to a
friendly whale icon on start-up.

Environment variables required:

* ``DISCORD_TOKEN`` – the token for your Discord bot.
* ``OPENROUTER_API_KEY`` – your OpenRouter API key for the DeepSeek model.
* ``SUMMARY_CHANNEL_ID`` – (optional) the channel ID where daily summaries will be posted.

If a `.env` file is present in the working directory, the
``dotenv`` package will load variables from it automatically.

To run the bot, install the dependencies listed in the README and
execute ``python3 bot.py``.
"""

import os
import asyncio
import datetime
from collections import defaultdict, deque
from typing import Deque, Dict, List

import aiohttp
import discord
from discord.ext import commands, tasks

# Load environment variables from a .env file if present
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except ImportError:
    pass


def get_env(var: str, default: str | None = None) -> str:
    """Helper to get environment variables and raise errors if missing."""
    value = os.getenv(var, default)
    if value is None:
        raise RuntimeError(f"Environment variable {var} is required but not set.")
    return value


async def call_llm(messages: List[Dict[str, str]], api_key: str, *, stream: bool = False) -> str:
    """
    Send a chat completion request to the OpenRouter API using the DeepSeek free model.
    """
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload: Dict[str, object] = {
        "model": "deepseek/deepseek-chat-v3-0324:free",
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 1024,
    }
    if stream:
        payload["stream"] = True
    timeout = aiohttp.ClientTimeout(total=60)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(url, headers=headers, json=payload) as resp:
            resp.raise_for_status()
            data = await resp.json()
            return data["choices"][0]["message"]["content"]


class DeepSeekBot(commands.Bot):
    def __init__(self) -> None:
        intents = discord.Intents.default()
        intents.message_content = True
        intents.members = True
        super().__init__(command_prefix="!", intents=intents, description="DeepSeek LLM Powered Bot")

        # Conversation history per channel (up to 15 messages)
        self.histories: Dict[int, Deque[Dict[str, str]]] = defaultdict(lambda: deque(maxlen=15))
        self.api_key: str = get_env("OPENROUTER_API_KEY")

        # Add commands
        self.add_command(self.summarize)  # type: ignore
        self.add_command(self.randomfact)  # type: ignore
        self.add_command(self.kick_member)  # type: ignore
        self.add_command(self.ban_member)  # type: ignore
        self.add_command(self.mute_member)  # type: ignore

    async def on_ready(self) -> None:
        print(f"Logged in as {self.user} (ID: {self.user.id})")
        await self.change_presence(activity=discord.Game(name="Exploring the deep!"))

        avatar_path = os.path.join(os.path.dirname(__file__), "deepseek_avatar.png")
        if os.path.exists(avatar_path):
            try:
                with open(avatar_path, "rb") as f:
                    avatar_bytes = f.read()
                await self.user.edit(avatar=avatar_bytes)
            except Exception as e:
                print(f"Could not update avatar: {e}")

    async def setup_hook(self) -> None:
        self.daily_summary.start()

    async def on_message(self, message: discord.Message) -> None:
        if message.author.bot:
            return

        # Save history
        history = self.histories[message.channel.id]
        history.append({"role": "user", "content": message.content})

        should_respond = False
        prompt = message.content
        if self.user.mentioned_in(message):
            prompt = prompt.replace(self.user.mention, "").strip()
            should_respond = True
        elif message.content.startswith("!ask"):
            prompt = message.content[len("!ask") :].strip()
            should_respond = True

        if should_respond:
            # ✅ SYSTEM PROMPT ADDED HERE
            system_prompt = (
                "You are QueryAsk, a helpful and creative AI assistant for Discord. You MUST provide engaging, witty, friendly, likable, empathetic, kind responses. Use slang and abbrievations where appropriate to lighten the mood and vibe with the user. Be casual where appropriate. Use slang. Use abbreviations and casual, gen-Z, gen-Alpha language WHERE appropriate, but maintain a cool, respectful, intelligent, helpful and honest tone. "
                "Always be friendly, concise, and thoughtful. Avoid illegal or harmful suggestions. "
                "Encourage curiosity and fun conversation."
            )

            context: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
            context.extend(list(history))
            context.append({"role": "user", "content": prompt})

            try:
                async with message.channel.typing():
                    response = await call_llm(context, self.api_key)
                await message.reply(response)
                history.append({"role": "assistant", "content": response})
            except Exception as e:
                await message.channel.send(f"An error occurred while contacting DeepSeek: {e}")

        await self.process_commands(message)

    # ================== Command Definitions ==================
    @commands.command(name="summarize", help="Summarize the last N messages in the channel")
    async def summarize(self, ctx: commands.Context, limit: int = 20) -> None:
        messages: List[str] = []
        async for msg in ctx.channel.history(limit=limit):
            if not msg.author.bot and msg.content:
                messages.append(f"{msg.author.display_name}: {msg.content}")
        if not messages:
            await ctx.send("No messages to summarise.")
            return
        text = "\n".join(reversed(messages))
        prompt_messages = [
            {"role": "system", "content": "Summarise the following Discord conversation in a concise paragraph."},
            {"role": "user", "content": text},
        ]
        try:
            async with ctx.typing():
                summary = await call_llm(prompt_messages, self.api_key)
            await ctx.send(f"**Summary:**\n{summary}")
        except Exception as e:
            await ctx.send(f"Error generating summary: {e}")

    @commands.command(name="randomfact", help="Get a random interesting fact")
    async def randomfact(self, ctx: commands.Context) -> None:
        prompt_messages = [
            {"role": "system", "content": "You are DeepSeek, a friendly AI."},
            {"role": "user", "content": "Tell me one random interesting fact about science or history."},
        ]
        try:
            async with ctx.typing():
                fact = await call_llm(prompt_messages, self.api_key)
            await ctx.send(fact)
        except Exception as e:
            await ctx.send(f"Failed to retrieve a fact: {e}")

    @commands.command(name="kick")
    @commands.has_permissions(kick_members=True)
    async def kick_member(self, ctx: commands.Context, member: discord.Member, *, reason: str | None = None) -> None:
        try:
            await member.kick(reason=reason)
            await ctx.send(f"{member.mention} has been kicked. Reason: {reason or 'No reason provided.'}")
        except Exception as e:
            await ctx.send(f"Failed to kick member: {e}")

    @commands.command(name="ban")
    @commands.has_permissions(ban_members=True)
    async def ban_member(self, ctx: commands.Context, member: discord.Member, *, reason: str | None = None) -> None:
        try:
            await member.ban(reason=reason)
            await ctx.send(f"{member.mention} has been banned. Reason: {reason or 'No reason provided.'}")
        except Exception as e:
            await ctx.send(f"Failed to ban member: {e}")

    @commands.command(name="mute")
    @commands.has_permissions(manage_roles=True)
    async def mute_member(self, ctx: commands.Context, member: discord.Member, minutes: int = 10) -> None:
        if minutes <= 0:
            await ctx.send("Duration must be positive.")
            return
        guild = ctx.guild
        if guild is None:
            return
        muted_role = discord.utils.get(guild.roles, name="Muted")
        if muted_role is None:
            try:
                muted_role = await guild.create_role(name="Muted", reason="Created by DeepSeek bot")
                for channel in guild.channels:
                    await channel.set_permissions(muted_role, speak=False, send_messages=False, add_reactions=False)
            except Exception as e:
                await ctx.send(f"Failed to create Muted role: {e}")
                return
        try:
            await member.add_roles(muted_role, reason=f"Muted by {ctx.author} for {minutes} minutes")
            await ctx.send(f"{member.mention} has been muted for {minutes} minute(s).")
        except Exception as e:
            await ctx.send(f"Failed to mute member: {e}")
            return

        async def unmute_after_delay():
            await asyncio.sleep(minutes * 60)
            try:
                await member.remove_roles(muted_role, reason="Mute duration expired")
                await ctx.channel.send(f"{member.mention} has been unmuted.")
            except Exception as e:
                await ctx.channel.send(f"Failed to unmute member: {e}")

        self.loop.create_task(unmute_after_delay())

    @tasks.loop(hours=24)
    async def daily_summary(self) -> None:
        channel_id_str = os.getenv("SUMMARY_CHANNEL_ID")
        if not channel_id_str:
            return
        try:
            channel_id = int(channel_id_str)
        except ValueError:
            return
        channel = self.get_channel(channel_id)
        if not isinstance(channel, discord.TextChannel):
            return
        now = datetime.datetime.utcnow()
        yesterday = now - datetime.timedelta(days=1)
        messages: List[str] = []
        async for msg in channel.history(after=yesterday, limit=2000):
            if not msg.author.bot and msg.content:
                messages.append(f"{msg.author.display_name}: {msg.content}")
        if not messages:
            await channel.send("It was quiet here today. Nothing to summarise!")
            return
        text = "\n".join(reversed(messages))
        prompt_messages = [
            {"role": "system", "content": "Summarise the following Discord conversation for a daily report."},
            {"role": "user", "content": text},
        ]
        try:
            summary = await call_llm(prompt_messages, self.api_key)
            await channel.send(f"**Daily Summary:**\n{summary}")
        except Exception as e:
            await channel.send(f"Failed to generate daily summary: {e}")

    @daily_summary.before_loop
    async def before_daily_summary(self) -> None:
        await self.wait_until_ready()


def main() -> None:
    token = get_env("DISCORD_TOKEN")
    bot = DeepSeekBot()
    bot.run(token)


if __name__ == "__main__":
    main()
