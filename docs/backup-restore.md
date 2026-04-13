# Backup and Restore

## Critical Files

The following files are stored on the Docker volume (`agent-assist-data`) and must be backed up:

| File | Description | Impact of Loss |
|------|-------------|----------------|
| `/data/.fernet_key` | Encryption key for all secrets | ALL encrypted secrets unrecoverable |
| `/data/agent_assist.db` | SQLite database (settings, traces, conversations) | All configuration and history lost |
| `/data/chromadb/` | Vector database (entity index, cache) | Rebuilt on next entity sync; cache lost |

## Backup Procedure

### Option 1: Volume Backup

```bash
# Stop the container
docker compose stop agent-assist

# Create backup
docker run --rm -v agent-assist-data:/data -v $(pwd)/backup:/backup alpine \
    tar czf /backup/agent-assist-backup-$(date +%Y%m%d).tar.gz /data

# Restart
docker compose start agent-assist
```

### Option 2: Fernet Key Export

Export the Fernet key via the admin API:

```bash
curl -s http://localhost:8080/api/admin/fernet-key-backup \
  -H "Cookie: <session_cookie>" | jq .key
```

Store this key in a secure location (password manager, vault).

## Restore Procedure

1. Stop the container.
2. Extract the backup into the Docker volume.
3. Start the container and verify via `/api/health`.

## Key Rotation (Advanced)

Currently, key rotation requires:
1. Export the current Fernet key.
2. Decrypt all secrets with the old key.
3. Replace `/data/.fernet_key` with a new key.
4. Re-encrypt all secrets with the new key.
5. Restart the container.
