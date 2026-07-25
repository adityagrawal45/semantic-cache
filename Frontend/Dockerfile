# Minimal static file server for the Recall playground frontend.
# No build step needed — it's a single dependency-free HTML file.
FROM nginx:1.27-alpine

COPY index.html /usr/share/nginx/html/index.html

# nginx's default config already serves index.html on / and handles
# static assets correctly; nothing custom needed for a single-page file.

EXPOSE 80

HEALTHCHECK --interval=15s --timeout=5s --start-period=5s --retries=5 \
    CMD wget -q -O /dev/null http://localhost:80/ || exit 1