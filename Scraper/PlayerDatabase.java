import java.sql.*;

public class PlayerDatabase {
    private static final String DB_URL = "jdbc:sqlite:vlr_players.db";

    public static void main(String[] args) {
        try (Connection conn = DriverManager.getConnection(DB_URL)) {
            if (conn != null) {
                Statement stmt = conn.createStatement();
                ResultSet rs = stmt.executeQuery("SELECT * FROM players ORDER BY name");
                System.out.println("ID | Name | Team | Rating | ACS | K/D | KAST | ADR | KPR | APR | FKPR | FDPR | HS% | Clutch%");
                while (rs.next()) {
                    System.out.printf("%d | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s\n",
                        rs.getInt("id"),
                        rs.getString("name"),
                        rs.getString("team"),
                        rs.getString("rating"),
                        rs.getString("average_combat_score"),
                        rs.getString("kill_deaths"),
                        rs.getString("kill_assists_survived_traded"),
                        rs.getString("average_damage_per_round"),
                        rs.getString("kills_per_round"),
                        rs.getString("assists_per_round"),
                        rs.getString("first_kills_per_round"),
                        rs.getString("first_deaths_per_round"),
                        rs.getString("headshot_percentage"),
                        rs.getString("clutch_success_percentage")
                    );
                }
            }
        } catch (SQLException e) {
            System.out.println(e.getMessage());
        }
    }
} 