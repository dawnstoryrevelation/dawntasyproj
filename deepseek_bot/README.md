# DeepSeek Discord Bot

**DeepSeek** is an autonomous, fully‑featured Discord bot built around the `deepseek/deepseek‑chat‑v3‑0324:free` language model hosted on [OpenRouter](https://openrouter.ai/deepseek/deepseek-chat-v3-0324:free).  
The bot can chat with members using the powerful DeepSeek LLM, summarise conversations, generate random facts on demand, and perform basic server administration (kick, ban, mute, etc.) when authorised by server administrators.

## Features

* **Conversational chat** – mention the bot or use `!ask` followed by your question and it will respond using the DeepSeek language model.  
  Conversation history is preserved for each channel so replies stay context‑aware.
* **Slash‑like commands** (prefix commands in this implementation):
  * `!summarize [N]` – summarises the last `N` messages (default: 20) in the current channel using the LLM.
  * `!randomfact` – asks the LLM to tell a random interesting fact.
  * `!kick @user [reason]`, `!ban @user [reason]` – remove misbehaving members (requires appropriate Discord permissions).
  * `!mute @user [minutes]` – temporarily mutes a user by adding a muted role (see below).
* **Daily summaries** – an optional scheduled task posts a daily summary of the previous day’s conversation to a channel of your choice.
* **Avatar** – uses a friendly blue whale icon inspired by DeepSeek’s whale logo.  If you wish to customise the avatar, replace `deepseek_avatar.png` with your preferred image.

## Requirements

* Python 3.9 or later.
* Discord bot token with **Message Content Intent** and **Server Members Intent** enabled.  
  See the [Discord Developer Portal](https://discord.com/developers/applications) to create an application, add a bot, enable intents and invite it to your server.
* An **OpenRouter API key**.  Although the DeepSeek model has a free tier, OpenRouter still requires a key for authentication.  
  Follow the instructions on OpenRouter to create a free account and generate an API key.

### Environment Variables

The bot reads several values from environment variables.  You can create a `.env` file or export them in your shell before running:

| Variable            | Purpose                                                                                   |
|---------------------|-------------------------------------------------------------------------------------------|
| `DISCORD_TOKEN`     | Your Discord bot token (keep this secret!).                                               |
| `OPENROUTER_API_KEY`| Your OpenRouter API key (for the DeepSeek free model).                                    |
| `SUMMARY_CHANNEL_ID`| *(Optional)* channel ID where daily summaries should be posted (integer).                  |

## Installation

1. Clone or copy this repository onto a machine where you intend to run the bot.
2. Install the required Python packages:

```bash
python3 -m pip install --upgrade pip
python3 -m pip install discord.py aiohttp python-dotenv
```

3. Place the avatar image `deepseek_avatar.png` in the root of the `deepseek_bot` folder.  A default whale icon is provided.
4. Set up your environment variables (`DISCORD_TOKEN`, `OPENROUTER_API_KEY`, and optionally `SUMMARY_CHANNEL_ID`).  For example:

```bash
export DISCORD_TOKEN="your-discord-bot-token"
export OPENROUTER_API_KEY="your-openrouter-key"
export SUMMARY_CHANNEL_ID="123456789012345678"
```

5. Run the bot:

```bash
python3 bot.py
```

## Adding the Bot to Your Server

After creating the bot in the Discord Developer Portal, you must generate an invite link with the proper permissions.  
At minimum, the bot needs the following OAuth scopes and permissions:

* **bot** scope
* **applications.commands** scope (for slash commands if you add them later)
* Permissions: `Send Messages`, `Read Messages`, `Manage Messages`, `Kick Members`, `Ban Members`, `Manage Roles` (for muting), and any others relevant to your desired features.

Use the URL generator on the Developer Portal to craft the invite link with these permissions and add the bot to your server.

## Muted Role

The `!mute` command assigns a role called **Muted** to the targeted user for a specified number of minutes.  You must create this role in your server with no speaking permissions and position it appropriately so that it overrides channel permissions.  If the role doesn’t exist, the bot will attempt to create it automatically.

## Limitations and Notes

* **API usage** – The DeepSeek model is accessed via OpenRouter’s API, which mandates an API key even for the free tier.  The [DEV article on building a free chatbot](https://dev.to/web_dev-usman/here-how-to-build-a-chatbot-for-free-using-openrouter-and-deepseek-apis-492e) explains that you must create an API key and include it in the request header【235835447775026†L168-L175】.  The same article provides an example `fetch` call that sends `Authorization: \`Bearer <API_KEY>\`` and uses `model: "deepseek/deepseek-r1:free"`【235835447775026†L202-L216】.  This bot adapts that call for the `deepseek/deepseek-chat-v3-0324:free` model.
* **Rate limits** – Free usage is subject to OpenRouter’s rate limits.  If you receive HTTP 429 errors, you may need to wait before making additional requests.
* **Privacy** – Conversation history is stored in memory only for context.  Restarting the bot clears this history.

## Extending the Bot

This bot is intentionally modular.  To add new commands or behaviours, examine `bot.py` and follow the patterns used for existing commands.  You can also implement slash commands using `discord.app_commands.CommandTree` if you prefer Discord’s new interactions over prefix commands.

Have fun exploring the depths with DeepSeek!