/**
 * Data models for the sample JS app.
 */

class BaseModel {
  toDict() {
    return { ...this };
  }

  validate() {
    return true;
  }
}

class User extends BaseModel {
  constructor(username, email, passwordHash) {
    super();
    this.username = username;
    this.email = email;
    this.passwordHash = passwordHash;
    this.isActive = true;
    this.role = 'user';
  }
}

class AdminUser extends User {
  constructor(username, email, passwordHash) {
    super(username, email, passwordHash);
    this.role = 'admin';
  }

  canDelete(resource) {
    return this.isActive;
  }
}

module.exports = { BaseModel, User, AdminUser };
