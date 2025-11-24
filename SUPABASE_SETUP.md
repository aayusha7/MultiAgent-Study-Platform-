# 🗄️ Supabase Integration Setup Guide

This guide will help you set up Supabase for **persistent data storage** so your analytics and RL data never get lost, even on Streamlit Cloud free tier.

## ✅ Why Supabase?

- **Free tier**: 500MB database, 2GB bandwidth/month
- **Truly persistent**: Data never gets cleared
- **PostgreSQL**: Industry-standard database
- **Easy setup**: 5-10 minutes
- **Works with Streamlit Cloud**: Perfect for deployment

## 📋 Step-by-Step Setup

### Step 1: Create Supabase Account

1. Go to [supabase.com](https://supabase.com)
2. Click **"Start your project"** or **"Sign up"**
3. Sign up with GitHub (recommended) or email
4. Verify your email if needed

### Step 2: Create a New Project

1. Click **"New Project"**
2. Fill in:
   - **Name**: `multiagent-study-platform` (or any name)
   - **Database Password**: Create a strong password (save it!)
   - **Region**: Choose closest to you
3. Click **"Create new project"**
4. Wait 2-3 minutes for project to initialize

### Step 3: Create Database Tables

1. In your Supabase project, go to **SQL Editor** (left sidebar)
2. Click **"New query"**
3. Copy and paste the contents of `supabase_setup.sql` from this repo
4. Click **"Run"** (or press Cmd/Ctrl + Enter)
5. You should see: "Success. No rows returned"

### Step 4: Get Your Credentials

1. Go to **Settings** → **API** (left sidebar)
2. You'll see:
   - **Project URL**: `https://xxxxx.supabase.co`
   - **anon/public key**: `eyJhbGc...` (long string)
   - **service_role key**: `eyJhbGc...` (keep this secret!)

**For Streamlit Cloud:**
- Use the **anon/public key** (safer for client-side)
- Or use **service_role key** if you need full access (more powerful but less secure)

### Step 5: Add to Streamlit Cloud Secrets

1. Go to your app on [share.streamlit.io](https://share.streamlit.io)
2. Click **"⋮"** (three dots) → **"Settings"**
3. Go to **"Secrets"** tab
4. Add these secrets:

```toml
SUPABASE_URL = "https://your-project-id.supabase.co"
SUPABASE_KEY = "your-anon-key-here"
```

**Example:**
```toml
SUPABASE_URL = "https://abcdefghijklmnop.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImFiY2RlZmdoaWprbG1ub3AiLCJyb2xlIjoiYW5vbiIsImlhdCI6MTYzODk2NzI4MCwiZXhwIjoxOTU0NTQzMjgwfQ.xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

5. Click **"Save"**

### Step 6: Redeploy Your App

1. In Streamlit Cloud, click **"Reboot app"** or push a new commit
2. Wait for deployment to complete
3. Your app will now use Supabase! 🎉

## 🔍 Verify It's Working

1. **Register a new user** in your app
2. **Answer some quiz questions**
3. **Check Supabase Dashboard**:
   - Go to **Table Editor** → **users** (should see your user)
   - Go to **Table Editor** → **rl_state** (should see your analytics data)

## 📊 What Gets Saved

With Supabase, these are **permanently saved**:

- ✅ **User accounts** - Never lost
- ✅ **RL learning state** - Personalization persists
- ✅ **Analytics data** - All quiz/flashcard performance
- ✅ **File mappings** - PDF filename tracking
- ✅ **User preferences** - Survey results, mode preferences

## 🔧 Troubleshooting

### "Supabase not configured" in logs
- ✅ Check that `SUPABASE_URL` and `SUPABASE_KEY` are in Streamlit Secrets
- ✅ Verify the keys are correct (no extra spaces)
- ✅ Make sure you saved the secrets

### "Table does not exist" error
- ✅ Run the SQL from `supabase_setup.sql` in Supabase SQL Editor
- ✅ Check table names match exactly

### "Permission denied" error
- ✅ Use **anon/public key** for basic operations
- ✅ Or use **service_role key** for full access (more powerful)

### Data not appearing
- ✅ Check Supabase Dashboard → Table Editor
- ✅ Verify tables were created successfully
- ✅ Check app logs for errors

## 🔐 Security Notes

- **anon/public key**: Safe to use in client-side code (Streamlit)
- **service_role key**: Full database access - **NEVER expose in frontend**
- For Streamlit Cloud, **anon key is recommended** (safer)

## 📈 Free Tier Limits

Supabase free tier includes:
- ✅ 500MB database storage
- ✅ 2GB bandwidth/month
- ✅ Unlimited API requests
- ✅ Automatic backups

**This is plenty for most applications!**

## 🚀 Next Steps

Once Supabase is set up:
1. Your data persists **forever** (even on free tier)
2. Analytics and RL learning work across sessions
3. Users can log in from anywhere
4. Ready for production! 🎉

---

**Need help?** Check the Supabase docs: [supabase.com/docs](https://supabase.com/docs)

