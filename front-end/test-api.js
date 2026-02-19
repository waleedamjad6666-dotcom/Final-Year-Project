// Test script to verify API connectivity
async function testAPI() {
  const baseURL = 'http://127.0.0.1:8000/api/v1';
  
  try {
    console.log('Testing API Health Check...');
    const healthResponse = await fetch(`${baseURL}/health/`);
    const healthData = await healthResponse.json();
    console.log('Health Check Result:', healthData);
    
    if (healthResponse.ok) {
      console.log('✅ Backend API is running and accessible!');
      
      // Test user registration
      console.log('\nTesting User Registration...');
      const registerData = {
        first_name: 'Test',
        last_name: 'User',
        email: `test${Date.now()}@example.com`,
        password: 'testpass123',
        password_confirm: 'testpass123'
      };
      
      const registerResponse = await fetch(`${baseURL}/auth/register/`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(registerData)
      });
      
      const registerResult = await registerResponse.json();
      
      if (registerResponse.ok) {
        console.log('✅ User Registration Successful:', registerResult);
        
        // Test login
        console.log('\nTesting User Login...');
        const loginResponse = await fetch(`${baseURL}/auth/login/`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            email: registerData.email,
            password: registerData.password
          })
        });
        
        const loginResult = await loginResponse.json();
        
        if (loginResponse.ok) {
          console.log('✅ User Login Successful:', loginResult);
          console.log('Token:', loginResult.token);
          
          // Test authenticated endpoint
          console.log('\nTesting Authenticated Endpoint...');
          const quotaResponse = await fetch(`${baseURL}/videos/quota/`, {
            headers: {
              'Authorization': `Token ${loginResult.token}`
            }
          });
          
          if (quotaResponse.ok) {
            const quotaData = await quotaResponse.json();
            console.log('✅ Authenticated Request Successful:', quotaData);
          } else {
            console.log('❌ Authenticated Request Failed:', quotaResponse.status);
          }
        } else {
          console.log('❌ Login Failed:', loginResult);
        }
      } else {
        console.log('❌ Registration Failed:', registerResult);
      }
    } else {
      console.log('❌ Backend API is not accessible');
    }
  } catch (error) {
    console.error('❌ API Test Failed:', error.message);
    console.log('\nMake sure the Django backend is running on http://127.0.0.1:8000/');
  }
}

// Run the test
testAPI();