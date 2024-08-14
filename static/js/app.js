const chatButton = document.querySelector('.chatbox__button');
const chatContent = document.querySelector('.chatbox__support');
const icons = {
    isClicked: '</p>Clicked!</p>',
    isNotClicked: '<p>Not clicked!</p>'
}
const chatbox = new InteractiveChatbox(chatButton, chatContent, icons);
chatbox.display();
chatbox.toggleIcon(false, chatButton);

// Get chatbot elements
const chatbot = document.getElementById('chatbot');
const conversation = document.getElementById('conversation');
const inputForm = document.getElementById('input-form');
const inputField = document.getElementById('input-field');

// Add event listener to input form
inputForm.addEventListener('submit', function(event) {
  // Prevent form submission
  event.preventDefault();

  // Get user input
  const input = inputField.value;

  // Clear input field
  inputField.value = '';
  const currentTime = new Date().toLocaleTimeString([], { hour: '2-digit', minute: "2-digit" });

  // Add user input to conversation
  let message = document.createElement('div');
  message.classList.add('messages__item', 'messages__item--visitor');
  message.innerHTML = `${input}`;
//   message.innerHTML = `<p class="chatbot-text" sentTime="${currentTime}">${input}</p>`;
  conversation.appendChild(message);

  // Generate chatbot response
  generateResponse(input);
//   const response = generateResponse(input);
//   console.log("response: ", response);

});

// Generate chatbot response function
function generateResponse(input) {
    $.get("/get/chatbot", { prediction_input: input }).done(function (msg) {
        console.log(input);
        console.log(msg);
        const response = msg;
        console.log("response: ",response);
        
        // Add chatbot response to conversation
        message = document.createElement('div');
        message.classList.add('messages__item','messages__item--operator');
        message.innerHTML = `${response}`;
        conversation.appendChild(message);
        message.scrollIntoView({behavior: "smooth"});
        
        // return msgText;
        // appendMessage(BOT_NAME, BOT_IMG, "left", msgText);
    });

    // Add chatbot logic here
    // const responses = [
    //   "Hello, how can I help you today? 😊",
    //   "I'm sorry, I didn't understand your question. Could you please rephrase it? 😕",
    //   "I'm here to assist you with any questions or concerns you may have. 📩",
    //   "I'm sorry, I'm not able to browse the internet or access external information. Is there anything else I can help with? 💻",
    //   "What would you like to know? 🤔",
    //   "I'm sorry, I'm not programmed to handle offensive or inappropriate language. Please refrain from using such language in our conversation. 🚫",
    //   "I'm here to assist you with any questions or problems you may have. How can I help you today? 🚀",
    //   "Is there anything specific you'd like to talk about? 💬",
    //   "I'm happy to help with any questions or concerns you may have. Just let me know how I can assist you. 😊",
    //   "I'm here to assist you with any questions or problems you may have. What can I help you with today? 🤗",
    //   "Is there anything specific you'd like to ask or talk about? I'm here to help with any questions or concerns you may have. 💬",
    //   "I'm here to assist you with any questions or problems you may have. How can I help you today? 💡",
    // ];
    
    // // Return a random response
    // return responses[Math.floor(Math.random() * responses.length)];
  }