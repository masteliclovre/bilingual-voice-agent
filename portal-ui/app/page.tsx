export default function Home() {
  return (
    <div>
      <h1>Voice Agent Portal</h1>
      <p>
        Go to <a href="/overview">Overview</a> to view KPIs.
      </p>
      <p>
        <a href="/api/auth/login">Log in</a> or{" "}
        <a href="/api/auth/logout">log out</a>.
      </p>
    </div>
  );
}
