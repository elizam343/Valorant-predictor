<?php
$db = new SQLite3('vlr_players.db');
$result = $db->query('SELECT * FROM players ORDER BY name');
?>
<!DOCTYPE html>
<html>
<head>
    <title>VLR Players</title>
    <style>
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ccc; padding: 8px; text-align: left; }
        th { background: #eee; }
    </style>
</head>
<body>
    <h1>VLR Players</h1>
    <table>
        <tr>
            <th>ID</th><th>Name</th><th>Team</th><th>Rating</th><th>ACS</th><th>K/D</th><th>KAST</th><th>ADR</th><th>KPR</th><th>APR</th><th>FKPR</th><th>FDPR</th><th>HS%</th><th>Clutch%</th>
        </tr>
        <?php while ($row = $result->fetchArray(SQLITE3_ASSOC)): ?>
        <tr>
            <td><?= htmlspecialchars($row['id']) ?></td>
            <td><?= htmlspecialchars($row['name']) ?></td>
            <td><?= htmlspecialchars($row['team']) ?></td>
            <td><?= htmlspecialchars($row['rating']) ?></td>
            <td><?= htmlspecialchars($row['average_combat_score']) ?></td>
            <td><?= htmlspecialchars($row['kill_deaths']) ?></td>
            <td><?= htmlspecialchars($row['kill_assists_survived_traded']) ?></td>
            <td><?= htmlspecialchars($row['average_damage_per_round']) ?></td>
            <td><?= htmlspecialchars($row['kills_per_round']) ?></td>
            <td><?= htmlspecialchars($row['assists_per_round']) ?></td>
            <td><?= htmlspecialchars($row['first_kills_per_round']) ?></td>
            <td><?= htmlspecialchars($row['first_deaths_per_round']) ?></td>
            <td><?= htmlspecialchars($row['headshot_percentage']) ?></td>
            <td><?= htmlspecialchars($row['clutch_success_percentage']) ?></td>
        </tr>
        <?php endwhile; ?>
    </table>
</body>
</html> 