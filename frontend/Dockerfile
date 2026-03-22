# Use official Node.js image
FROM node:20-alpine

# Set working directory
WORKDIR /app

# Copy package files
COPY package.json package-lock.json ./

# Install dependencies
RUN npm install

# Copy all frontend code
COPY . .

# Expose Next.js port
EXPOSE 3000

# Start development server by default
CMD ["npm", "run", "dev"]
