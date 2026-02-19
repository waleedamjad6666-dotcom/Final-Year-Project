# Video Translation Backend API

A Django REST API backend for video translation and processing with user authentication.

## Features

### Authentication & User Management
- ✅ User Registration with email/password
- ✅ JWT Authentication (access & refresh tokens)
- ✅ User Login/Logout with token blacklisting
- ✅ User Profile Management
- ✅ Password Change
- ✅ Account Deletion
- ✅ User Storage Tracking
- ✅ Subscription Tiers (Free, Basic, Premium, Enterprise)

### Video Management
- ✅ Video Upload (MP4, AVI, MOV, MKV, WEBM)
- ✅ Video Metadata Storage
- ✅ Video Processing Status Tracking
- ✅ Processing Progress Monitoring
- ✅ Processing Logs
- ✅ Video Download
- ✅ User Video Statistics
- ✅ Video CRUD Operations

## Tech Stack

- **Django 5.2.8** - Web framework
- **Django REST Framework** - REST API
- **Simple JWT** - JWT authentication
- **Django CORS Headers** - CORS support
- **Python Decouple** - Environment configuration
- **Pillow** - Image processing
- **SQLite** - Database (can be changed to PostgreSQL/MySQL)

## Installation

1. **Navigate to back-end directory:**
   ```bash
   cd back-end
   ```

2. **Activate virtual environment:**
   ```bash
   # Windows
   .\venv\Scripts\Activate.ps1
   
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables:**
   - Copy `.env.example` to `.env` (if needed)
   - Update the values in `.env` file

5. **Run migrations:**
   ```bash
   python manage.py migrate
   ```

6. **Create superuser (optional):**
   ```bash
   python manage.py createsuperuser
   ```

7. **Run development server:**
   ```bash
   python manage.py runserver
   ```

The API will be available at `http://127.0.0.1:8000/`

## API Endpoints

### Authentication Endpoints

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| POST | `/api/accounts/register/` | Register new user | No |
| POST | `/api/accounts/login/` | Login user | No |
| POST | `/api/accounts/logout/` | Logout user | Yes |
| POST | `/api/accounts/token/refresh/` | Refresh access token | No |

### User Profile Endpoints

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| GET/PUT | `/api/accounts/profile/` | Get/Update profile | Yes |
| PUT/PATCH | `/api/accounts/profile/update/` | Update user info | Yes |
| POST | `/api/accounts/password/change/` | Change password | Yes |
| DELETE | `/api/accounts/delete/` | Delete account | Yes |

### Video Endpoints

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| GET/POST | `/api/videos/` | List/Create videos | Yes |
| GET/PUT/DELETE | `/api/videos/{id}/` | Video details | Yes |
| GET | `/api/videos/{id}/download/` | Download video | Yes |
| GET | `/api/videos/{id}/status/` | Processing status | Yes |
| GET | `/api/videos/{id}/logs/` | Processing logs | Yes |
| GET | `/api/videos/stats/user/` | User statistics | Yes |

## Request/Response Examples

### Register User
```bash
POST /api/accounts/register/
Content-Type: application/json

{
  "username": "johndoe",
  "email": "john@example.com",
  "password": "securepass123",
  "password2": "securepass123",
  "first_name": "John",
  "last_name": "Doe",
  "phone_number": "+1234567890"
}
```

### Login
```bash
POST /api/accounts/login/
Content-Type: application/json

{
  "email": "john@example.com",
  "password": "securepass123"
}
```

### Upload Video
```bash
POST /api/videos/
Authorization: Bearer {access_token}
Content-Type: multipart/form-data

{
  "title": "My Video",
  "description": "Video description",
  "original_video": <file>,
  "source_language": "en",
  "target_language": "es"
}
```

## Environment Variables

Create a `.env` file in the `back-end` directory:

```env
SECRET_KEY=your-secret-key-here
DEBUG=True
ALLOWED_HOSTS=localhost,127.0.0.1

# Database
DB_NAME=db.sqlite3

# CORS
CORS_ALLOWED_ORIGINS=http://localhost:3000,http://localhost:5173

# JWT Settings
JWT_ACCESS_TOKEN_LIFETIME=60
JWT_REFRESH_TOKEN_LIFETIME=1440
```

## Project Structure

```
back-end/
├── config/               # Project settings
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
├── accounts/             # Authentication & users
│   ├── models.py
│   ├── serializers.py
│   ├── views.py
│   ├── urls.py
│   └── admin.py
├── videos/               # Video management
│   ├── models.py
│   ├── serializers.py
│   ├── views.py
│   ├── urls.py
│   └── admin.py
├── media/                # Uploaded files
├── venv/                 # Virtual environment
├── manage.py
├── requirements.txt
├── .env
└── .gitignore
```

## Models

### User Model
- Custom user model with email as username
- Additional fields: phone, profile picture, bio, date of birth
- Email verification status

### UserProfile Model
- Storage tracking (used/limit)
- Video count
- Subscription tier
- Last login IP

### Video Model
- UUID primary key
- File storage for original and processed videos
- Processing status and progress
- Language settings
- Metadata (duration, size, resolution)

### ProcessingLog Model
- Step-by-step processing logs
- Log levels (info, warning, error)
- Timestamps

## Admin Panel

Access the admin panel at `http://127.0.0.1:8000/admin/`

Features:
- User management
- Video management
- Processing logs viewer
- User profiles management

## Next Steps (AI Features)

The following AI features will be implemented:
1. Video transcription (speech-to-text)
2. Text translation
3. Text-to-speech (TTS)
4. Lip-sync generation
5. Video processing pipeline

## Development Notes

- All API endpoints (except register/login) require JWT authentication
- Include `Authorization: Bearer {access_token}` header in requests
- Tokens expire after configured time (default: 60 min for access, 24h for refresh)
- File uploads limited to 100MB per video
- Supported video formats: MP4, AVI, MOV, MKV, WEBM

## Testing

You can test the API using:
- **Postman** or **Insomnia**
- **curl** commands
- Django REST Framework browsable API at `http://127.0.0.1:8000/api/`

## License

MIT License
