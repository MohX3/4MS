# GitHub Setup Instructions

This guide will help you connect your local repository to GitHub and set up collaboration for your team.

## Step 1: Create a New Repository on GitHub

1. Go to [GitHub](https://github.com) and sign in
2. Click the "+" icon in the top right corner
3. Select "New repository"
4. Name your repository (e.g., `TalentTalk`)
5. **Do NOT** initialize with README, .gitignore, or license (since we already have these)
6. Click "Create repository"

## Step 2: Connect Your Local Repository to GitHub

After creating the repository, GitHub will show you commands. Use these commands in your terminal:

```bash
# Add the remote repository (replace YOUR_USERNAME and REPO_NAME with your actual values)
git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git

# Or if you prefer SSH (requires SSH key setup):
# git remote add origin git@github.com:YOUR_USERNAME/REPO_NAME.git

# Verify the remote was added
git remote -v
```

## Step 3: Push Your Code to GitHub

```bash
# Push to the main branch
git push -u origin main
```

If you encounter authentication issues:
- For HTTPS: GitHub will prompt for credentials. You may need to use a Personal Access Token instead of your password
- For SSH: Make sure your SSH key is added to your GitHub account

## Step 4: Set Up Team Collaboration

### For Team Members (Initial Setup)

Each team member should:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/REPO_NAME.git
   cd REPO_NAME
   ```

2. **Set up environment variables:**
   ```bash
   # Copy the example file
   cp .env.example .env
   
   # Edit .env and add your API key
   # (Each team member should use their own API key or a shared team key)
   ```

3. **Install dependencies:**
   ```bash
   # Create virtual environment
   python -m venv venv
   
   # Activate virtual environment
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   
   # Install packages
   pip install -r requirements.txt
   ```

### Branch Protection (Optional but Recommended)

To protect your main branch and require code reviews:

1. Go to your repository on GitHub
2. Click "Settings" → "Branches"
3. Under "Branch protection rules", click "Add rule"
4. Enter `main` as the branch name pattern
5. Enable:
   - ✅ Require a pull request before merging
   - ✅ Require approvals (set to 1 or more)
   - ✅ Require status checks to pass before merging (if you set up CI/CD)
6. Click "Create"

## Step 5: Recommended Workflow for Team

### Creating a Feature Branch

```bash
# Make sure you're on main and up to date
git checkout main
git pull origin main

# Create and switch to a new branch
git checkout -b feature/your-feature-name

# Make your changes, then commit
git add .
git commit -m "Description of your changes"

# Push the branch to GitHub
git push -u origin feature/your-feature-name
```

### Creating a Pull Request

1. After pushing your branch, go to GitHub
2. You'll see a banner suggesting to create a pull request
3. Click "Compare & pull request"
4. Fill in the description of your changes
5. Request reviews from team members
6. Once approved, merge the pull request

### Updating Your Local Repository

```bash
# Fetch latest changes
git fetch origin

# Switch to main branch
git checkout main

# Pull latest changes
git pull origin main
```

## Important Notes

- **Never commit sensitive data**: The `.env` file is gitignored, but always double-check before committing
- **Use meaningful commit messages**: Describe what and why, not just what
- **Keep branches focused**: One feature or fix per branch
- **Pull before pushing**: Always pull latest changes before pushing to avoid conflicts
- **Communicate with your team**: Let team members know about major changes

## Troubleshooting

### If you get "remote origin already exists"
```bash
# Remove existing remote
git remote remove origin

# Add the correct remote
git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git
```

### If you need to update the remote URL
```bash
git remote set-url origin https://github.com/YOUR_USERNAME/REPO_NAME.git
```

### If you have merge conflicts
```bash
# Pull with rebase to avoid merge commits
git pull --rebase origin main

# Resolve conflicts in your editor, then:
git add .
git rebase --continue
```

