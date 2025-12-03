#!/bin/bash
# Fix My Bag CSS placement in all HTML files

echo "Moving My Bag CSS to <head> section in all HTML files..."

FILES="public/user.html public/achievements.html public/profile.html"

for FILE in $FILES; do
    echo "Processing $FILE..."
    
    # Check if file exists
    if [ ! -f "$FILE" ]; then
        echo "  ⚠️  File not found: $FILE"
        continue
    fi
    
    # Check if CSS link is already in head
    if grep -q '<link rel="stylesheet" href="/static/css/my-bag.css">' "$FILE" | head -20; then
        echo "  ✅ CSS already in head"
    else
        # Add CSS link after sidebar.css in head
        sed -i '' '/<link rel="stylesheet" href="\/static\/sidebar.css">/a\
    <link rel="stylesheet" href="/static/css/my-bag.css">
' "$FILE"
        echo "  ✅ Added CSS to head"
    fi
    
    # Remove CSS link from body (if exists)
    if grep -q '<link rel="stylesheet" href="/static/css/my-bag.css">' "$FILE" | tail -n +50; then
        # Find and remove the standalone link tag in body
        sed -i '' '/<body>/,/<\/body>/ {
            /<link rel="stylesheet" href="\/static\/css\/my-bag.css">/d
        }' "$FILE"
        echo "  ✅ Removed duplicate from body"
    fi
done

echo ""
echo "✅ All files processed!"
