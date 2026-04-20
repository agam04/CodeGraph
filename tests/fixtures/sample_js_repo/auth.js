/**
 * Authentication utilities for the sample JS app.
 */

const crypto = require('crypto');

/**
 * Hash a password using SHA-256.
 * @param {string} password - Plaintext password.
 * @returns {string} Hex digest.
 */
function hashPassword(password) {
  return crypto.createHash('sha256').update(password).digest('hex');
}

/**
 * Authenticate a user against the user store.
 * @param {string} username
 * @param {string} password
 * @param {Object} users - Map of username to user objects.
 * @returns {Object|null} Session object or null.
 */
async function authenticate(username, password, users) {
  const user = users[username];
  if (!user) return null;
  const hash = hashPassword(password);
  if (hash !== user.passwordHash) return null;
  return createSession(user);
}

function createSession(user) {
  return {
    token: crypto.randomBytes(16).toString('hex'),
    user,
    expiresAt: Date.now() + 3600 * 1000,
  };
}

function invalidateSession(session) {
  session.expiresAt = 0;
  return true;
}

module.exports = { hashPassword, authenticate, createSession, invalidateSession };
