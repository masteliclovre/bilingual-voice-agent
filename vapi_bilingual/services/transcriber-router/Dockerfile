FROM node:20-alpine
WORKDIR /app
COPY package.json package-lock.json* pnpm-lock.yaml* yarn.lock* ./
RUN npm ci --include=dev || npm install --include=dev
COPY . .
RUN npm run build
EXPOSE 3001
CMD ["npm","start"]