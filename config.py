import os

BOT_CONFIG = {
    'api_id': int(os.getenv('TELEGRAM_API_ID', '38276360')),
    'api_hash': os.getenv('TELEGRAM_API_HASH', '29b3a8fd85db1590cc6cc12d9864be54'),
    'bot_token': os.getenv('BOT_TOKEN', '8561523022:AAEivpyrPjP7E-44bqObPKboatsM5YfX3tw'),
    'owner_id': int(os.getenv('OWNER_ID', '5812817910')),
    'mongo_uri': os.getenv('MONGO_URI', 'mongodb+srv://aztech:ayazahmed1122@cluster0.mhuaw3q.mongodb.net/goadsbot_db?retryWrites=true&w=majority'),
    'db_name': os.getenv('MONGO_DB_NAME', 'goadsbot_db'),
    'logger_bot_token': os.getenv('LOGGER_BOT_TOKEN', '8137849431:AAH3Kl3nRnn_sTne0-nSv9j1r8Xa-c8g7mc'),
    'logger_bot_username': os.getenv('LOGGER_BOT_USERNAME', 'goadslogbot'),
    # Admin Notifications Config
    'notification_bot_token': os.getenv('NOTIFICATION_BOT_TOKEN', '8137849431:AAH3Kl3nRnn_sTne0-nSv9j1r8Xa-c8g7mc'),
    'notification_channel_id': int(os.getenv('NOTIFICATION_CHANNEL_ID', '-1001234567890')),
}

# ===================== PLAN TIERS =====================
# Scout (Free), Grow (₹69), Prime (₹199), Dominion (₹389)

PLAN_SCOUT = {
    'name': 'Scout',
    'price': 0,
    'price_display': 'Free',
    'tagline': 'Perfect for beginners exploring automation',
    'emoji': '🔰',
    'max_accounts': 1,
    'msg_delay': 60,
    'round_delay': 900,
    'auto_reply_enabled': False,
    'max_topics': 2,
    'max_groups_per_topic': 10,
    'logs_enabled': False,
    'description': '1 account, slow delays (60s/900s), basic features',
}

PLAN_GROW = {
    'name': 'Grow',
    'price': 69,
    'price_display': '₹69',
    'tagline': 'Scale your reach with multiple accounts',
    'emoji': '📈',
    'max_accounts': 3,
    'msg_delay': 30,
    'round_delay': 600,
    'auto_reply_enabled': True,
    'max_topics': 5,
    'max_groups_per_topic': 50,
    'logs_enabled': True,
    'description': '3 accounts, medium delays (30s/600s), auto-reply + logs + 🔄 Smart Rotation + 👥 Auto Group Join',
}

PLAN_PRIME = {
    'name': 'Prime',
    'price': 199,
    'price_display': '₹199',
    'tagline': 'Advanced automation for serious marketers',
    'emoji': '⭐',
    'max_accounts': 7,
    'msg_delay': 10,
    'round_delay': 120,
    'auto_reply_enabled': True,
    'max_topics': 9,
    'max_groups_per_topic': 100,
    'logs_enabled': True,
    'description': '7 accounts, fast delays (10s/120s), full features + 🔄 Smart Rotation + 👥 Auto Group Join',
}

PLAN_DOMINION = {
    'name': 'Dominion',
    'price': 389,
    'price_display': '₹389',
    'tagline': 'Ultimate power for advertising domination',
    'emoji': '👑',
    'max_accounts': 15,
    'msg_delay': 10,
    'round_delay': 120,
    'auto_reply_enabled': True,
    'max_topics': 15,
    'max_groups_per_topic': 200,
    'logs_enabled': True,
    'description': '15 accounts, fastest delays (10s/120s), priority support + 🔄 Smart Rotation + 👥 Auto Group Join',
}

PLANS = {
    'scout': PLAN_SCOUT,
    'grow': PLAN_GROW,
    'prime': PLAN_PRIME,
    'dominion': PLAN_DOMINION,
}

# Backwards compat (old code references FREE_TIER/PREMIUM_TIER)
FREE_TIER = PLAN_SCOUT.copy()
PREMIUM_TIER = PLAN_DOMINION.copy()
ADMIN_USERNAME = "BlazeNXT"

MESSAGES = {
    'welcome': "Welcome to GO ADS BOT!\n\nManage your Telegram advertising campaigns with ease.",
    'welcome_image': os.getenv('WELCOME_IMAGE', 'https://i.ibb.co/84M4v9tP/start.jpg'),

    # ===================== Account Profile Templates =====================
    # Applied to ALL added accounts when user opens dashboard (/start).
    # First name is preserved as-is.
    # Last name is forced to this tag (removes any existing last name).
    'account_last_name_tag': '| @GoadsROBOT',
    # Bio is forced to this text (removes any existing bio).
    'account_bio': 'Smart Ads Automation • @GoadsROBOT',
    'support_link': os.getenv('SUPPORT_LINK', 'https://t.me/BlazeNXT'),
    'updates_link': os.getenv('UPDATES_LINK', 'https://t.me/goadsupdate'),
    'premium_contact': "Contact admin to purchase Premium access.\n\nPremium Benefits:\n- More accounts\n- Faster delays\n- Auto-reply feature\n- Detailed logs\n- Priority support",
    
    # Privacy Policy
    'privacy_short': (
        "<b>📜 Privacy Policy & Terms of Service</b>\n\n"
        "<blockquote>By using GO ADS BOT, you acknowledge and agree to:\n\n"
        "<b>✓ Service Usage:</b>\n"
        "• Automated broadcasting across Telegram groups\n"
        "• Responsible and ethical use of the platform\n"
        "• Compliance with Telegram's Terms of Service\n\n"
        "<b>✓ Data & Privacy:</b>\n"
        "• Session data stored securely (encrypted)\n"
        "• Account credentials never shared\n"
        "• Analytics for service improvement only\n"
        "• No data sold to third parties\n\n"
        "<b>✓ Your Responsibility:</b>\n"
        "• Avoid spam or abusive content\n"
        "• Respect group rules and user privacy\n"
        "• Use reasonable delays between messages</blockquote>\n\n"
        "<i>We prioritize your security and privacy.</i>"
    ),
    'privacy_full_link': os.getenv('PRIVACY_URL', 'https://example.com/privacy-policy'),
}

# ===================== Force Join (Config-based) =====================
# If enabled, users must join BOTH a channel and a group before using the bot.
# Use usernames (without @) so buttons can point to public links.
FORCE_JOIN = {
    'enabled': os.getenv('FORCE_JOIN_ENABLED', 'true').lower() in ('1', 'true', 'yes', 'on'),

    # Public @usernames (without @). Example: 'AdsReachUpdates'
    'channel_username': os.getenv('FORCE_JOIN_CHANNEL', 'goadsupdate'),
    # group_username removed (no forced group join)

    # Lock screen visuals
    'image_url': os.getenv('FORCE_JOIN_IMAGE', 'https://i.ibb.co/84M4v9tP/start.jpg'),
    'message': os.getenv(
        'FORCE_JOIN_MESSAGE',
        "**Access Locked**\n\nPlease join our **Channel** and **Group** to use this bot.\n\nAfter joining, click **Verify**."
    ),
}

# Plan-specific images (one for each plan)
PLAN_IMAGES = {
    'grow': os.getenv('GROW_PLAN_IMAGE', 'https://i.ibb.co/84M4v9tP/start.jpg'),
    'prime': os.getenv('PRIME_PLAN_IMAGE', 'https://i.ibb.co/84M4v9tP/start.jpg'),
    'dominion': os.getenv('DOMINION_PLAN_IMAGE', 'https://i.ibb.co/84M4v9tP/start.jpg'),
}

# ===================== Payment Config =====================
# Manual UPI payment (no crypto)
UPI_PAYMENT = {
    'qr_image_url': os.getenv('UPI_QR_IMAGE_URL', 'https://i.ibb.co/XZSzZbgC/qr.jpg'),
    'upi_id': os.getenv('UPI_ID', 'example@upi'),
    'payee_name': os.getenv('UPI_PAYEE_NAME', 'GO ADS BOT'),
}

INTERVAL_PRESETS = {
    'slow': {'msg_delay': 60, 'round_delay': 900, 'name': 'Slow (Safe)'},
    'medium': {'msg_delay': 30, 'round_delay': 600, 'name': 'Medium (Balanced)'},
    'fast': {'msg_delay': 10, 'round_delay': 120, 'name': 'Fast (Risky)'},
}

TOPICS = ['instagram', 'exchange', 'twitter', 'telegram', 'minecraft', 'tiktok', 'youtube', 'whatsapp', 'other']

